from astropy.table import Table

from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1

from astropy.coordinates import SkyCoord
import astropy.units as u

if __name__ == '__main__':
    positive_set = Table.read(
        "/home/duncan/PycharmProjects/MyResearchProject_Duncan/data/crossmatch/PS_positive_samples.csv", format="csv")
    negative_sample_path = "data/crossmatch/negative samples"
    # n_sample_csvs = os.listdir(negative_sample_path)
    # n_sample_csvs = [file_name[:-4] for file_name in n_sample_csvs]

    # VLASS = list(positive_set['VLASS_component_name'])
    # need_download = list(set(VLASS) - set(n_sample_csvs))
    df_p = positive_set.to_pandas()
    df_p.set_index(['VLASS_component_name'], inplace=True)
    count = 0

    negative_df = []

    for i, row in enumerate(positive_set):
        print(i)
        ra, dec = row['VLASS_RA'], row['VLASS_DEC']
        result_tables = Vizier.query_region(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), radius=78 * u.arcsec,
                                            catalog='II/349')
        candidate_tab: Table = result_tables[0]
        ids = list(candidate_tab['objID'])
        if not ids.__contains__(row['Pan-STARRS_objID']):
            print(f"No positive sample around {row['VLASS_component_name']} within 78 arcsecs")
            raise ValueError
        else:
            positive_row_index = ids.index(row['Pan-STARRS_objID'])
            candidate_tab.remove_row(positive_row_index)
            df_n = candidate_tab.to_pandas()
            df_n = df_n[['objID', 'RAJ2000', 'DEJ2000']]
            df_n.insert(0, "VLASS_RA", row['VLASS_RA'])
            df_n.insert(1, "VLASS_DEC", row['VLASS_DEC'])
            df_n.insert(2, "VLASS_E_RA", row['VLASS_E_RA'])
            df_n.insert(3, "VLASS_E_DEC", row['VLASS_E_DEC'])

            df_n.rename(
                columns={'objID': 'Pan-STARRS_objID', 'RAJ2000': 'Pan-STARRS_RAJ2000', 'DEJ2000': 'Pan-STARRS_DEJ2000'},
                inplace=True)

            negative_df.append(df_n)
