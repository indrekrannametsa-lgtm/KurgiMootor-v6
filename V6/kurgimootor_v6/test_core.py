import tempfile
from pathlib import Path
from core import KurgiDB

with tempfile.TemporaryDirectory() as td:
    db = KurgiDB(Path(td) / 'test.db')
    assert len(db.all_fields()) == 14

    day = '2026-08-03'
    db.ensure_default_plan(day)
    assert [r['field_id'] for r in db.plan_for(day)] == [1, 2, 3]

    db.replace_plan(day, [])
    db.ensure_default_plan(day)
    assert db.plan_for(day) == [], 'Tühi päev ei tohi automaatselt uuesti täituda'

    db.replace_plan(day, [4, 5, 6, 7])
    assert len(db.plan_for(day)) == 4

    db.save_harvest('2026-08-03', 4, 5, 1, 2, 3, 4)
    assert db.harvest_for('2026-08-03', 4)['interval_days'] == 5
    assert db.interval_days(4, '2026-08-04', 5) == 1

    n = db.import_weather('[{"date":"2026-08-01","t_avg":18.4,"radiation":21.7}]')
    assert n == 1
    assert db.weather_status() == (1, 0, 0)

print('Kõik põhitestid läbitud.')
