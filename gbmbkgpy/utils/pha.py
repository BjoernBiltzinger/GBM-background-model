import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import os
import warnings

from gbmbkgpy.io.fits_file import FITSExtension, FITSFile


def _atleast_2d_with_dtype(value, dtype=None):
    if dtype is not None:
        value = np.array(value, dtype=dtype)

    arr = np.atleast_2d(value)

    return arr


def _atleast_1d_with_dtype(value, dtype=None):
    if dtype is not None:
        value = np.array(value, dtype=dtype)

        if dtype == str:
            # convert None to NONE
            # which is needed for None Type args
            # to string arrays

            idx = np.core.defchararray.lower(value) == 'none'

            value[idx] = 'NONE'

    arr = np.atleast_1d(value)

    return arr


class SPECTRUM(FITSExtension):
    _HEADER_KEYWORDS = (('EXTNAME', 'SPECTRUM', 'Extension name'),
                        ('CONTENT', 'OGIP PHA data', 'File content'),
                        ('HDUCLASS', 'OGIP    ', 'format conforms to OGIP standard'),
                        ('HDUVERS', '1.1.0   ', 'Version of format (OGIP memo CAL/GEN/92-002a)'),
                        ('HDUDOC', 'OGIP memos CAL/GEN/92-002 & 92-002a', 'Documents describing the forma'),
                        ('HDUVERS1', '1.0.0   ', 'Obsolete - included for backwards compatibility'),
                        ('HDUVERS2', '1.1.0   ', 'Obsolete - included for backwards compatibility'),
                        ('HDUCLAS1', 'SPECTRUM', 'Extension contains spectral data  '),
                        ('HDUCLAS2', 'TOTAL ', ''),
                        ('HDUCLAS3', 'RATE ', ''),
                        ('HDUCLAS4', 'TYPE:II ', ''),
                        ('FILTER', '', 'Filter used'),
                        ('CHANTYPE', 'PHA', 'Channel type'),
                        ('POISSERR', False, 'Are the rates Poisson distributed'),
                        ('DETCHANS', None, 'Number of channels'),
                        ('CORRSCAL', 1.0, ''),
                        ('AREASCAL', 1.0, '')

                        )

    def __init__(self, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=None, sys_err=None, stat_err=None, is_poisson=False):

        """
        Represents the SPECTRUM extension of a PHAII file.

        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        n_spectra = len(tstart)

        data_list = [('TSTART', tstart),
                     ('TELAPSE', telapse),
                     ('SPEC_NUM', np.arange(1, n_spectra + 1, dtype=np.int16)),
                     ('CHANNEL', channel),
                     ('RATE', rate),
                     ('QUALITY', quality),
                     ('BACKSCAL', backscale),
                     ('GROUPING', grouping),
                     ('EXPOSURE', exposure),
                     ('RESPFILE', respfile),
                     ('ANCRFILE', ancrfile)]

        if back_file is not None:
            data_list.append(('BACKFILE', back_file))

        if stat_err is not None:
            assert is_poisson == False, "Tying to enter STAT_ERR error but have POISSERR set true"

            data_list.append(('STAT_ERR', stat_err))

        if sys_err is not None:
            data_list.append(('SYS_ERR', sys_err))

        super(SPECTRUM, self).__init__(tuple(data_list), self._HEADER_KEYWORDS)

        self.hdu.header.set("POISSERR", is_poisson)


class PHAII(FITSFile):

    def __init__(self, instrument_name, telescope_name, tstart, telapse, channel, rate, quality, grouping, exposure, backscale, respfile,
                 ancrfile, back_file=None, sys_err=None, stat_err=None, is_poisson=False):

        """

        A generic PHAII fits file

        :param instrument_name: name of the instrument
        :param telescope_name: name of the telescope
        :param tstart: array of interval start times
        :param telapse: array of times elapsed since start
        :param channel: arrary of channel numbers
        :param rate: array of rates
        :param quality: array of OGIP quality values
        :param grouping: array of OGIP grouping values
        :param exposure: array of exposures
        :param backscale: array of backscale values
        :param respfile: array of associated response file names
        :param ancrfile: array of associate ancillary file names
        :param back_file: array of associated background file names
        :param sys_err: array of optional systematic errors
        :param stat_err: array of optional statistical errors (required of non poisson!)
        """

        # collect the data so that we can have a general
        # extension builder

        self._tstart = _atleast_1d_with_dtype(tstart, np.float32) * u.s
        self._telapse = _atleast_1d_with_dtype(telapse, np.float32) * u.s
        self._channel = _atleast_2d_with_dtype(channel, np.int16)
        self._rate = _atleast_2d_with_dtype(rate, np.float32) * 1. / u.s
        self._exposure = _atleast_1d_with_dtype(exposure, np.float32) * u.s
        self._quality = _atleast_2d_with_dtype(quality, np.int16)
        self._grouping = _atleast_2d_with_dtype(grouping, np.int16)
        self._backscale = _atleast_1d_with_dtype(backscale, np.float32)
        self._respfile = _atleast_1d_with_dtype(respfile, str)
        self._ancrfile = _atleast_1d_with_dtype(ancrfile, str)

        if sys_err is not None:

            self._sys_err = _atleast_2d_with_dtype(sys_err, np.float32)

        else:

            self._sys_err = sys_err

        if stat_err is not None:

            self._stat_err = _atleast_2d_with_dtype(stat_err, np.float32)

        else:

            self._stat_err = stat_err

        if back_file is not None:

            self._back_file = _atleast_1d_with_dtype(back_file, str)
        else:

            self._back_file = np.array(['NONE'] * self._tstart.shape[0])

        # Create the SPECTRUM extension

        spectrum_extension = SPECTRUM(self._tstart,
                                      self._telapse,
                                      self._channel,
                                      self._rate,
                                      self._quality,
                                      self._grouping,
                                      self._exposure,
                                      self._backscale,
                                      self._respfile,
                                      self._ancrfile,
                                      back_file=self._back_file,
                                      sys_err=self._sys_err,
                                      stat_err=self._stat_err,
                                      is_poisson=is_poisson)

        # Set telescope and instrument name

        spectrum_extension.hdu.header.set("TELESCOP", telescope_name)
        spectrum_extension.hdu.header.set("INSTRUME", instrument_name)
        spectrum_extension.hdu.header.set("DETCHANS", len(self._channel[0]))

        super(PHAII, self).__init__(fits_extensions=[spectrum_extension])

    @classmethod
    def from_time_series(cls, time_series, use_poly=False):

        pha_information = time_series.get_information_dict(use_poly)

        is_poisson = True

        if use_poly:
            is_poisson = False

        return PHAII(instrument_name=pha_information['instrument'],
                     telescope_name=pha_information['telescope'],
                     tstart=pha_information['tstart'],
                     telapse=pha_information['telapse'],
                     channel=pha_information['channel'],
                     rate=pha_information['rates'],
                     stat_err=pha_information['rate error'],
                     quality=pha_information['quality'].to_ogip(),
                     grouping=pha_information['grouping'],
                     exposure=pha_information['exposure'],
                     backscale=1.,
                     respfile=None,  # pha_information['response_file'],
                     ancrfile=None,
                     is_poisson=is_poisson)

    @classmethod
    def from_fits_file(cls, fits_file):

        with fits.open(fits_file) as f:

            if 'SPECTRUM' in f:
                spectrum_extension = f['SPECTRUM']
            else:
                warnings.warn("unable to find SPECTRUM extension: not OGIP PHA!")

                spectrum_extension = None

                for extension in f:
                    hduclass = extension.header.get("HDUCLASS")
                    hduclas1 = extension.header.get("HDUCLAS1")

                    if hduclass == 'OGIP' and hduclas1 == 'SPECTRUM':
                        spectrum_extension = extension
                        warnings.warn("File has no SPECTRUM extension, but found a spectrum in extension %s" % (spectrum_extension.header.get("EXTNAME")))
                        spectrum_extension.header['EXTNAME'] = 'SPECTRUM'
                        break

            spectrum = FITSExtension.from_fits_file_extension(spectrum_extension)

            out = FITSFile(primary_hdu=f['PRIMARY'], fits_extensions=[spectrum])

        return out

    @property
    def instrument(self):
        return
