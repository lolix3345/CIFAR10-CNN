def convert_seconds_to_hms(seconds):
    h = int(seconds / 3600)
    seconds = seconds - h*3600
    m = int(seconds / 60)
    seconds = seconds - m*60
    s = int(seconds)
    return (h, m, s)
