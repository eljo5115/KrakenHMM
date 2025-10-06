"""Kraken API websocket client utilities.

This module provides a lightweight Kraken WebSocket client for receiving
ticker/trade data for given asset pairs. It uses `aiohttp` for websockets
and provides an async generator of price updates.
"""

from typing import AsyncGenerator, Dict, Any, List
import aiohttp
import asyncio
import json
import time

KRKN_WSS = "wss://ws.kraken.com/"


async def subscribe_pairs(session: aiohttp.ClientSession, pairs: List[str]):
    msg = {"event": "subscribe", "pair": pairs, "subscription": {"name": "ticker"}}
    await session.send_str(json.dumps(msg))


async def ticker_stream(pairs: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
    """Connect to Kraken websocket and yield ticker messages for given pairs.

    Yields dicts with at least 'pair' and 'price' keys.
    """
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(KRKN_WSS) as ws:
            await ws.send_json({"event": "subscribe", "pair": pairs, "subscription": {"name": "ticker"}})

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # ticker updates come as arrays, skip heartbeat/events
                    if isinstance(data, dict):
                        # event message
                        continue
                    # example: [channelID, {"a": [...], "b": [...], ...}, "ticker", "PAIR"]
                    try:
                        pair = data[-1]
                        payload = data[1]
                        # last trade price is often in payload['c'][0]
                        price = float(payload.get("c", [None])[0]) if payload.get("c") else None
                        # attach a timestamp from local clock (websocket payloads don't always include one)
                        yield {"pair": pair, "price": price, "raw": payload, "time": int(time.time())}
                    except Exception:
                        continue
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break


async def rest_ticker_stream(pairs: List[str], interval: float = 5.0) -> AsyncGenerator[Dict[str, Any], None]:
    """Poll Kraken REST Ticker endpoint and yield price updates.

    This is a simple polling fallback (or alternative) to the websocket.
    It returns dicts with 'pair', 'price', and 'volume' (24h volume) when
    available. `interval` controls seconds between polls.
    """
    import aiohttp
    import time

    # map normalized pair key used by Kraken REST: remove slash
    rest_keys = {p: p.replace("/", "") for p in pairs}
    url = "https://api.kraken.com/0/public/Ticker"

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                query = ",".join(rest_keys.values())
                async with session.get(url, params={"pair": query}, timeout=10) as resp:
                    data = await resp.json()
                result = data.get("result", {})
                # result keys may differ from requested (Kraken returns internal pair names)
                # build reverse map from returned key to original pair
                # we attempt to match by suffix
                for rkey, payload in result.items():
                    # find the original pair whose REST key is a substring of rkey
                    orig = None
                    for p, rk in rest_keys.items():
                        if rk in rkey or rkey in rk:
                            orig = p
                            break
                    if orig is None:
                        # fallback: try exact match
                        for p, rk in rest_keys.items():
                            if rk == rkey:
                                orig = p
                                break
                    # parse price and volume if present
                    price = None
                    volume = None
                    try:
                        # last trade closed price
                        c = payload.get("c")
                        if c:
                            price = float(c[0])
                        v = payload.get("v")
                        if v and len(v) > 1:
                            # second element is 24h volume
                            volume = float(v[1])
                    except Exception:
                        pass
                    if orig:
                        yield {"pair": orig, "price": price, "volume": volume, "raw": payload, "time": int(time.time())}
            except Exception:
                # network or parse issue; just ignore this tick
                pass
            await asyncio.sleep(interval)


async def fetch_ohlc(pairs: List[str], interval: int = 1440, max_candles: int = 400) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch recent OHLC candles from Kraken REST for each pair.

    Returns a mapping original_pair -> list of candles in chronological order.
    Each candle is a dict: {"time": int, "close": float, "volume": float}
    """
    import aiohttp
    url = "https://api.kraken.com/0/public/OHLC"
    rest_keys = {p: p.replace("/", "") for p in pairs}
    results: Dict[str, List[Dict[str, Any]]] = {}
    async with aiohttp.ClientSession() as session:
        for p, rk in rest_keys.items():
            try:
                async with session.get(url, params={"pair": rk, "interval": interval}, timeout=15) as resp:
                    data = await resp.json()
                r = data.get("result", {})
                # Kraken returns an entry keyed by an internal pair name; grab it
                if not r:
                    results[p] = []
                    continue
                inner = next(iter(r.values()))
                # inner is list of candles [time, open, high, low, close, vwap, volume, count]
                candles = []
                # take last max_candles entries
                for row in inner[-max_candles:]:
                    try:
                        t = int(row[0])
                        close = float(row[4])
                        vol = float(row[6])
                        candles.append({"time": t, "close": close, "volume": vol})
                    except Exception:
                        continue
                # ensure chronological order (oldest first)
                results[p] = candles
            except Exception:
                results[p] = []
            # be polite to Kraken API
            await asyncio.sleep(0.35)
    return results


async def fetch_daily_close_volume(pairs: List[str], days: int = 365) -> Dict[str, Dict[str, List[float]]]:
        """Fetch daily OHLC candles for the past `days` days and return close & volume series.

        Returns a mapping:
            pair -> {"time": [int epoch], "close": [float], "volume": [float]}

        This is a thin wrapper around `fetch_ohlc` using a 1440-minute (daily)
        interval and will attempt to return up to `days` candles per pair.
        """
        # reuse the OHLC helper with daily interval (minutes)
        daily_interval = 1440
        max_candles = max(1, int(days))
        raw = await fetch_ohlc(pairs, interval=daily_interval, max_candles=max_candles)

        out: Dict[str, Dict[str, List[float]]] = {}
        for p, candles in raw.items():
                # candles are already chronological (oldest first)
                times = [c.get("time") for c in candles]
                closes = [c.get("close") for c in candles]
                vols = [c.get("volume") for c in candles]
                out[p] = {"time": times, "close": closes, "volume": vols}
        return out


async def place_market_order(pair: str, volume: float, side: str, api_key: str, api_secret: str, timeout: int = 15) -> Dict:
    """Place a market order via Kraken private API.

    This helper requires valid `api_key` and `api_secret` (base64 string as provided
    by Kraken). It returns the parsed JSON result on success or raises an Exception
    on failure. This is a thin wrapper over /0/private/AddOrder implementing the
    required signing (API-Sign) per Kraken docs.
    """
    import aiohttp
    import time
    import hashlib
    import hmac
    import base64
    import urllib.parse

    if not api_key or not api_secret:
        raise ValueError("API key and secret are required to place orders")

    url_path = "/0/private/AddOrder"
    url = "https://api.kraken.com" + url_path

    nonce = str(int(time.time() * 1000))
    data = {
        "nonce": nonce,
        "pair": pair.replace("/", ""),
        "type": side.lower(),
        "ordertype": "market",
        "volume": str(volume),
    }
    postdata = urllib.parse.urlencode(data)

    # Kraken signing: API-Sign = base64_encode(HMAC_SHA512(api_secret_decoded, path + SHA256(nonce + postdata)))
    sha256 = hashlib.sha256((nonce + postdata).encode("utf-8")).digest()
    message = url_path.encode("utf-8") + sha256
    secret_decoded = base64.b64decode(api_secret)
    signature = hmac.new(secret_decoded, message, hashlib.sha512)
    api_sign = base64.b64encode(signature.digest()).decode()

    headers = {
        "API-Key": api_key,
        "API-Sign": api_sign,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=postdata, headers=headers, timeout=timeout) as resp:
            resp_json = await resp.json()
    if resp_json.get("error"):
        raise Exception(f"Kraken API error: {resp_json.get('error')}")
    result = resp_json.get("result", {})
    # If Kraken returned txid(s), attempt to query order details for executed fills
    txids = result.get("txid")
    if txids:
        try:
            order_info = await query_orders(txids, api_key, api_secret, timeout=timeout)
            return {"add_result": result, "order_info": order_info}
        except Exception:
            # best-effort: return raw add order result if query fails
            return {"add_result": result}
    return {"add_result": result}


async def query_orders(txids, api_key: str, api_secret: str, timeout: int = 15) -> Dict:
    """Query order details for given txid(s) via /0/private/QueryOrders.

    txids may be a list of txid strings or a single string. Returns the
    parsed 'result' mapping from Kraken if successful.
    """
    import aiohttp
    import time
    import hashlib
    import hmac
    import base64
    import urllib.parse

    if not api_key or not api_secret:
        raise ValueError("API key and secret are required to query orders")

    url_path = "/0/private/QueryOrders"
    url = "https://api.kraken.com" + url_path

    nonce = str(int(time.time() * 1000))
    if isinstance(txids, (list, tuple)):
        txs = ",".join(txids)
    else:
        txs = str(txids)
    data = {"nonce": nonce, "txid": txs}
    postdata = urllib.parse.urlencode(data)

    sha256 = hashlib.sha256((nonce + postdata).encode("utf-8")).digest()
    message = url_path.encode("utf-8") + sha256
    secret_decoded = base64.b64decode(api_secret)
    signature = hmac.new(secret_decoded, message, hashlib.sha512)
    api_sign = base64.b64encode(signature.digest()).decode()

    headers = {
        "API-Key": api_key,
        "API-Sign": api_sign,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=postdata, headers=headers, timeout=timeout) as resp:
            resp_json = await resp.json()
    if resp_json.get("error"):
        raise Exception(f"Kraken API error: {resp_json.get('error')}")
    return resp_json.get("result", {})
