#ifndef __MEL_FEATURE_EXTRACTOR__
#define __MEL_FEATURE_EXTRACTOR__

#include "common.h"
#include "libmel.h"
#include <string>
#include <cstddef>


class MelFeatureExtractor
{
private:
	const int m_sampleRate;
	const int m_winlen;
	const int m_nfft;
	const int m_nhop;
	const bool m_center;
	const std::string m_mode;
	const float m_power;
	const int m_nmels;
	const float m_fmin;
	const float m_fmax;
	const Matrixf m_melfilter;
	const Vectorf m_window;
	const bool	  m_log;
	const NormalizeType m_normalize;
public:
	/// \brief      Create mel spectrogram feature extractor
/// \param      sampleRate            sample rate of 'x'
/// \param      n_fft         length of the FFT size
/// \param      n_hop         number of samples between successive frames
/// \param      window        window function. currently only supports 'hann'
/// \param      center        If True, the signal y is padded so that frame t is centered at y[t * hop_length]. If False, then frame t begins at y[t * hop_length]
///	\param      mode          pad mode. support "reflect","symmetric","edge"
/// \param      power         exponent for the magnitude melspectrogram
/// \param      n_mels        number of mel bands
/// \param      f_min         lowest frequency (in Hz)
/// \param      f_max         highest frequency (in Hz)
	MelFeatureExtractor(
		int sampleRate,
		int winlen,
		int n_fft,
		int n_hop,
		const std::string& window,
		bool center,
		const std::string& mode,
		float power,
		int n_mels,
		float f_min,
		float f_max,
		NormalizeType normalize,
		bool log);
	~MelFeatureExtractor();

	//estimate size of mel spectrogram based on size of X
	
	Matrixf Extract(const Vectorf &x);

	const size_t estimate_output_buffer_size(size_t samples);
};


#endif //__MEL_FEATURE_EXTRACTOR__

