function end_datenum = F_tai93_to_datenum(time_tai93,total_leap_seconds)
start_datenum = datenum([1993 1 1 0 0 0]);
end_datenum = start_datenum+(time_tai93-total_leap_seconds)/86400;