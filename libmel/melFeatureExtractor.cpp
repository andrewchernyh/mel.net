#include "common.h"
#include "utils.h"
#include "melFeatureExtractor.h"

MelFeatureExtractor::MelFeatureExtractor(
    int sampleRate,
    int winlen,
    int nfft,
    int nhop,
    const std::string& window,
    bool center,
    const std::string& mode,
    float power,
    int nmels,
    float fmin,
    float fmax,
    NormalizeType normalize,
    bool log)
    :m_sampleRate(sampleRate),
    m_nfft(nfft),
    m_nhop(nhop),
    m_window(createWindow(window, winlen, nfft)),
    m_center(center),
    m_mode(mode),
    m_power(power),
    m_nmels(nmels),
    m_fmin(fmin),
    m_fmax(fmax),
    m_winlen(winlen),
    m_melfilter(melfilter(sampleRate, nfft, nmels, fmin, fmax)),
    m_normalize(normalize),
    m_log(log)
{

}

MelFeatureExtractor::~MelFeatureExtractor()
{

}


//same as in nemo 
//https://github.com/NVIDIA/NeMo/blob/7b1e82c40c102ca77eaf75ed741822667d9d7ea8/nemo/collections/asr/parts/preprocessing/features.py#L51
const float std_const = 1E-5f;

const float log_zero_guard_value = 0.000000059604644775390625f;

static Matrixf log(Matrixf data)
{
    auto res = (data.array() + log_zero_guard_value).log();
    return res;
}



void NormalizePerFeature(Matrixf& mel)
{
    auto ones = Vectorf(mel.cols());
    ones.setConstant(1.0f);
    for (auto i = 0; i < mel.rows(); i++)
    {
        Vectorf row = mel.row(i);
        auto mean = row.mean();
        auto diff = row - (mean * ones);
        auto std = std::sqrt(diff.squaredNorm() / (row.size() - 1));
        mel.row(i) = diff / (std + std_const);
    }
}


Matrixf MelFeatureExtractor::Extract(const Vectorf &x)
{
    auto X = stft(x, m_window,  m_nfft, m_nhop, m_center, m_mode);
    auto sp = spectrogram(X, m_power);
    Matrixf mel = m_melfilter * sp.transpose();
    if (m_log)
        mel = log(mel);
    switch (m_normalize)
    {
        case PerFeature:
            NormalizePerFeature(mel);
            break;
        case None:
            break;
    }

    return mel;
}

const size_t MelFeatureExtractor::estimate_output_buffer_size(size_t samples)
{
    return ((samples / m_nhop) + 1) * m_nmels;
}