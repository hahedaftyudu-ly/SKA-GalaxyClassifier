from astropy.logger import Logger
from panstamps.downloader import downloader


def getFits(ra, dec):
    fitsPaths, jpegPaths, colorPath = downloader(
        log=Logger(name="duncan's research"),
        fits=True,
        jpeg=False,
        ra=ra,
        dec=dec,
        color=False,
        imageType='stack',
        filterSet='grizy'
    ).get()
    return fitsPaths


fitsPath = getFits(ra=147.46914, dec=0.61810944)
print(fitsPath)
# for i in range(tab_len):
#     fitsPath = getFits(T['ra'][i], T['dec'][i])
#     print(i)
