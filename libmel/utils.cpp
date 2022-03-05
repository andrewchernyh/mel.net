#include "utils.h"
#include <Eigen/Core>
#include <unsupported/Eigen/FFT>

static Vectorf pad(const Vectorf& x, int left, int right, const std::string& mode, float value) {
    Vectorf x_paded = Vectorf::Constant(left + x.size() + right, value);
    x_paded.segment(left, x.size()) = x;

    if (mode.compare("reflect") == 0) {
        for (int i = 0; i < left; ++i) {
            x_paded[i] = x[left - i];
        }
        for (int i = left; i < left + right; ++i) {
            x_paded[i + x.size()] = x[x.size() - 2 - i + left];
        }
    }

    if (mode.compare("symmetric") == 0) {
        for (int i = 0; i < left; ++i) {
            x_paded[i] = x[left - i - 1];
        }
        for (int i = left; i < left + right; ++i) {
            x_paded[i + x.size()] = x[x.size() - 1 - i + left];
        }
    }

    if (mode.compare("edge") == 0) {
        for (int i = 0; i < left; ++i) {
            x_paded[i] = x[0];
        }
        for (int i = left; i < left + right; ++i) {
            x_paded[i + x.size()] = x[x.size() - 1];
        }
    }

    if (mode.compare("constant") == 0) {
        for (int i = 0; i < left; ++i) {
            x_paded[i] = 0.0f;
        }
        for (int i = left; i < left + right; ++i) {
            x_paded[i + x.size()] = 0.0f;
        }
    }

    return x_paded;
}


Matrixf melfilter(int sr, int n_fft, int n_mels, float fmin, float fmax) {
    int n_f = n_fft / 2 + 1;
    Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f - 1)) * sr) / n_fft;

    float f_min = 0.f;
    float f_sp = 200.f / 3.f;
    float min_log_hz = 1000.f;
    float min_log_mel = (min_log_hz - f_min) / f_sp;
    float logstep = logf(6.4f) / 27.f;

    auto hz_to_mel = [=](float hz, bool htk = false) -> float {
        if (htk) {
            return 2595.0f * log10f(1.0f + hz / 700.0f);
        }
        float mel = (hz - f_min) / f_sp;
        if (hz >= min_log_hz) {
            mel = min_log_mel + logf(hz / min_log_hz) / logstep;
        }
        return mel;
    };
    auto mel_to_hz = [=](Vectorf& mels, bool htk = false) -> Vectorf {
        if (htk) {
            return 700.0f * (Vectorf::Constant(n_mels + 2, 10.f).array().pow(mels.array() / 2595.0f) - 1.0f);
        }
        return (mels.array() > min_log_mel).select(((mels.array() - min_log_mel) * logstep).exp() * min_log_hz, (mels * f_sp).array() + f_min);
    };

    float min_mel = hz_to_mel(fmin);
    float max_mel = hz_to_mel(fmax);
    Vectorf mels = Vectorf::LinSpaced(n_mels + 2, min_mel, max_mel);
    Vectorf mel_f = mel_to_hz(mels);
    Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
    Matrixf ramps = mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(n_mels + 2, 1).array();

    Matrixf lower = -ramps.topRows(n_mels).array() / fdiff.segment(0, n_mels).transpose().replicate(1, n_f).array();
    Matrixf upper = ramps.bottomRows(n_mels).array() / fdiff.segment(1, n_mels).transpose().replicate(1, n_f).array();
    Matrixf weights = (lower.array() < upper.array()).select(lower, upper).cwiseMax(0);

    auto enorm = (2.0 / (mel_f.segment(2, n_mels) - mel_f.segment(0, n_mels)).array()).transpose().replicate(1, n_f);
    weights = weights.array() * enorm;

    return weights;
}

Vectorf hann(int winlen, int n_fft)
{
    Vectorf window(n_fft);
    window.setZero();
    int pad = (n_fft - winlen) / 2;
    for (int i = 0; i < winlen; ++i)
        window[i + pad] = 0.5f * (1.0f - cos((2.f * EIGEN_PI * i)/(winlen-1)));
    return window;
}

Vectorf createWindow(const std::string& window, int winlen, int n_fft)
{
    if (window.compare("hann") == 0)
        return hann(winlen, n_fft);
    throw NotImplementedWindow();
}

Matrixcf stft(const Vectorf& x, const Vectorf &window, int n_fft, int n_hop, bool center, const std::string& mode) {
    int pad_len = center ? n_fft / 2 : 0;
    Vectorf x_paded = pad(x, pad_len, pad_len, mode, 0.f);

    int n_f = n_fft / 2 + 1;
    auto n_frames = 1 + (x_paded.size() - n_fft) / n_hop;
    Matrixcf X(n_frames, n_fft);
    Eigen::FFT<float> fft;

    for (int i = 0; i < n_frames; ++i) {
        Vectorf x_frame = window.array() * x_paded.segment(i * n_hop, n_fft).array();
        X.row(i) = fft.fwd(x_frame);
    }
    return X.leftCols(n_f);
}

Matrixf spectrogram(const Matrixcf& X, float power) {
    return X.cwiseAbs().array().pow(power);
}
