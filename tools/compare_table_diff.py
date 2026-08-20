from astropy.table import Table
from astroquery.sdss import SDSS

T = Table.read("SDSS_zwarning_cleaned.tbl", format="ipac")


def test_specID():
    for i in range(3638808, len(T)):
        r = T[i]
        query = "select s.specObjID from specObj as s where s.ra = " + str(r['ra']) + " and s.dec = " + str(r['dec'])
        data = SDSS.query_sql_async(query)
        if data.status_code == 200:
            strs = data.text.split("\n")
            print(f"i = {i}: " + strs[2])
        else:
            raise Exception


test_specID()
