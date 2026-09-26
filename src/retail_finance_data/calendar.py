"""Public holidays, computed rather than downloaded (the workspace has no internet)."""

from __future__ import annotations

from datetime import date, timedelta


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def public_holidays(country: str, years: list[int]) -> set[date]:
    """National public holidays on which physical stores close. DE and CH only."""
    out: set[date] = set()
    for y in years:
        e = easter_sunday(y)
        moving = [e - timedelta(days=2), e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)]
        if country == "DE":
            fixed = [date(y, 1, 1), date(y, 5, 1), date(y, 10, 3), date(y, 12, 25), date(y, 12, 26)]
        elif country == "CH":
            fixed = [date(y, 1, 1), date(y, 1, 2), date(y, 8, 1), date(y, 12, 25), date(y, 12, 26)]
        else:
            raise ValueError(f"unsupported country {country!r}")
        out.update(fixed + moving)
    return out
