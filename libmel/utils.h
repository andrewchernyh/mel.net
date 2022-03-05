#ifndef __UTILS_H__
#define __UTILS_H__
#include "common.h"
#include <string>
#include <stdexcept>

class NotImplementedWindow : public std::logic_error
{
public:
    NotImplementedWindow() : std::logic_error("Window not yet implemented") { };
};


Matrixf melfilter(int sr, int n_fft, int n_mels, float fmin, float fmax);
Matrixcf stft(const Vectorf& x, const Vectorf& window, int n_fft, int n_hop, bool center, const std::string& mode);

//creates window function array of winlen size, and then, padded to n_fft size to center
Vectorf createWindow(const std::string& window,int winlen, int n_fft);
Matrixf spectrogram(const Matrixcf& X, float power = 1.f);

#endif //__UTILS_H__

