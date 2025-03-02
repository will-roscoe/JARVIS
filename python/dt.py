from datetime import datetime
from random import randint

dates = [
    "2020-01-01T00:00:00",
    "2020-01-02T20:02:10",
    "2020-01-03T04:01:01"]

secofday = [[randint(0,86400) for i in range(4)],
            [randint(0,86400) for i in range(4)],
            [randint(0,86400) for i in range(4)]
            ]

for i, date in enumerate(dates):
    print(f"Date: {date}")
    for j in secofday[i]:
        hours = j // 3600
        seco = j % 3600
        minutes = seco // 60
        seco = seco % 60
        corrected = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
        corrected = corrected.replace(hour=hours, minute=minutes, second=seco)
        print(f"Seconds of day: {j} -> {corrected}")

        

        