import pytz
from datetime import datetime, timedelta


def get_current_ecommerce_event(current_date=None) -> str | None:
    """
    Determines the current e-commerce event based on the provided UTC date/time.
    If no date is provided, uses the current system date/time in UTC.
    Returns the event name or None if no event is active.
    
    Args:
        current_date (datetime, optional): The UTC date to check. Must be timezone-aware (UTC).
                                          Defaults to current UTC date/time.
    
    Returns:
        str or None: Name of the current e-commerce event, or None if no event is active.
    """
    # Ensure UTC timezone
    utc = pytz.UTC
    
    # If no date provided, use current UTC time
    if current_date is None:
        current_date = datetime.now(utc)
    
    # Ensure input date is UTC-aware
    if current_date.tzinfo is None:
        current_date = utc.localize(current_date)
    else:
        current_date = current_date.astimezone(utc)

    # Define e-commerce events with start and end dates for a typical year
    # Format: (event_name, start_date, end_date)
    # Dates are month/day, year-agnostic, with appropriate lead-up periods
    events = [
        # Valentine's Day: 2-week lead-up, ends on Feb 14
        ("Valentine's Day", 
         lambda y: utc.localize(datetime(y, 1, 31)), 
         lambda y: utc.localize(datetime(y, 2, 14, 23, 59, 59))),
        
        # Mother's Day: 2-week lead-up, ends on second Sunday of May
        ("Mother's Day", 
         lambda y: utc.localize(datetime(y, 5, 1)),  # Start May 1
         lambda y: utc.localize(datetime(y, 5, 8)) + timedelta(days=6)),  # Second Sunday
        
        # Father's Day: 2-week lead-up, ends on third Sunday of June
        ("Father's Day", 
         lambda y: utc.localize(datetime(y, 6, 1)),  # Start June 1
         lambda y: utc.localize(datetime(y, 6, 15)) + timedelta(days=6)),  # Third Sunday
        
        # Back to School: Mid-July to early September
        ("Back to School", 
         lambda y: utc.localize(datetime(y, 7, 15)), 
         lambda y: utc.localize(datetime(y, 9, 10, 23, 59, 59))),
        
        # Halloween: 2-week lead-up, ends on Oct 31
        ("Halloween", 
         lambda y: utc.localize(datetime(y, 10, 15)), 
         lambda y: utc.localize(datetime(y, 10, 31, 23, 59, 59))),
        
        # Black Friday: Promote only on the day (day after Thanksgiving, fourth Thursday of November)
        ("Black Friday", 
         lambda y: utc.localize(datetime(y, 11, 22)) + timedelta(days=2),  # Day after fourth Thursday
         lambda y: utc.localize(datetime(y, 11, 29, 23, 59, 59)) + timedelta(days=2)),
        
        # Cyber Monday: Promote only on the Monday after Black Friday
        ("Cyber Monday", 
         lambda y: utc.localize(datetime(y, 11, 22)) + timedelta(days=5),  # Monday after Black Friday
         lambda y: utc.localize(datetime(y, 11, 22, 23, 59, 59)) + timedelta(days=5)),
        
        # Christmas/Holiday Season: December 1 to December 25
        ("Christmas", 
         lambda y: utc.localize(datetime(y, 12, 1)), 
         lambda y: utc.localize(datetime(y, 12, 25, 23, 59, 59))),

        # Boxing Week: December 26 - December 30
         ("Boxing Day", 
         lambda y: utc.localize(datetime(y, 12, 26)), 
         lambda y: utc.localize(datetime(y, 12, 29, 23, 59, 59))),
        
        # New Year's Sales: December 29 to January 5
        ("New Year's Sales", 
         lambda y: utc.localize(datetime(y, 12, 30)),
         lambda y: utc.localize(datetime(y+1, 1, 5, 23, 59, 59)))
    ]

    current_year = current_date.year

    for event_name, start_date_func, end_date_func in events:
        start_date = start_date_func(current_year)
        end_date = end_date_func(current_year)

        # Handle events that span year-end (e.g., New Year's Sales)
        if end_date.year > current_year:
            if current_date >= start_date or current_date <= end_date.replace(year=current_date.year):
                return event_name
        # Normal case: event within the same year
        elif start_date <= current_date <= end_date:
            return event_name

    return None
