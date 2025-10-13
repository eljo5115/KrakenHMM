import asyncio
from kraken_hmm.trader import Trader

async def test():
    t = Trader(total_capital=1000, n_assets=1)
    # set a position
    class Pos:
        pass
    pos = Pos()
    pos.pair = 'AAA/BBB'
    pos.qty = 1.0
    pos.entry_price = 100.0
    pos.stop_price = 95.0
    pos.take_price = None
    t.positions['AAA/BBB'] = pos
    # simulate a tick that triggers stop (price <= 95)
    await t.on_tick('AAA/BBB', 94.0)
    print('last_exits after sell:', t._last_exits.get('AAA/BBB'))
    # now compute allocations where AAA/BBB would be missing, ensure re-entry blocked
    def fake_rank():
        return ['AAA/BBB']
    t.rank_assets = fake_rank
    allocs = t.allocate()
    print('allocs initial (allocate uses rank_assets):', allocs)
    # simulate price change but within cooldown
    await t.on_tick('AAA/BBB', 94.2)
    print('positions after attempt (should be None if blocked):', t.positions.get('AAA/BBB'))

asyncio.get_event_loop().run_until_complete(test())
