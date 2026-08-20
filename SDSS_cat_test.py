from astroquery.cadc import Cadc

if __name__ == '__main__':
    cadc = Cadc()
    for collection, details in sorted(cadc.get_collections().items()):
        print(f'{collection} : {details}')
