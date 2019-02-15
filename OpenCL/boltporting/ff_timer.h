#pragma once

//#include "ff_basic.h"
#include <time.h>

namespace feifei
{
	class TimerBase
	{
	protected:
		timespec startTime;
		timespec stopTime;

	public:
		virtual void Restart() = 0;
		virtual void Stop() = 0;

		double ElapsedMilliSec = 0;
		double ElapsedNanoSec = 0;
	};

#ifdef _WIN32
	class WinTimer
	{
	public:
		void Restart();
		void Stop();
	};
#endif

	class UnixTimer:public TimerBase
	{
	public:
		void Restart();
		void Stop();
	};


#ifdef _WIN32
	void WinTimer::Restart()
	{
		QueryPerformanceFrequency(&cpuFreqHz);
		QueryPerformanceCounter(&startTime);
	}

	void WinTimer::Stop()
	{
		double diffTime100ns;
		QueryPerformanceCounter(&stopTime);
		diffTime100ns = (stopTime.QuadPart - startTime.QuadPart) * 1000.0 / cpuFreqHz.QuadPart;
		ElapsedMilliSec = diffTime100ns / 10.0;
	}
#endif

	void UnixTimer::Restart()
	{
		clock_gettime(CLOCK_MONOTONIC, &startTime);
	}

	void UnixTimer::Stop()
	{
		float diffTime100ns;
		clock_gettime(CLOCK_MONOTONIC, &stopTime);
		double d_startTime = static_cast<double>(startTime.tv_sec)*1e9 + static_cast<double>(startTime.tv_nsec);
		double d_currentTime = static_cast<double>(stopTime.tv_sec)*1e9 + static_cast<double>(stopTime.tv_nsec);
		ElapsedNanoSec = d_currentTime - d_startTime;
		ElapsedMilliSec = ElapsedNanoSec / 1e6;
	}


}


class Timer2
{
public:
	void Start() { gettimeofday(&m_start, NULL); }
	void End() { gettimeofday(&m_end, NULL); }
	double  GetDelta()
	{
		return (m_end.tv_sec - m_start.tv_sec) * 1000.0
			+ (m_end.tv_usec - m_start.tv_usec) / 1000.0;
	}
private:
	struct timeval m_start;
	struct timeval m_end;
};
