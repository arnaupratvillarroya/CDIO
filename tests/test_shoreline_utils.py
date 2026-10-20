import datetime
from shoreline_utils import extract_date

def test_extract_date():
    assert extract_date("image_2021-06-11.tif") == datetime.date(2021, 6, 11)
    assert extract_date("data_20190523_raw.csv") == datetime.date(2019, 5, 23)
    assert extract_date("invalid_filename.txt") is None
