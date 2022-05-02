/*

Copyright(c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#pragma once

#include <cassert>
#include <limits>
#include <execution>


namespace aln
{
#ifndef ALN_DEBUG_FORCE_SEQUENTIAL

    constexpr auto ParallelUnsequenced = std::execution::par_unseq;
    constexpr auto Parallel = std::execution::par;
    constexpr auto Sequential = std::execution::seq;

#else

    constexpr auto ParallelUnsequenced = std::execution::seq;
    constexpr auto Parallel = std::execution::seq;
    constexpr auto Sequential = std::execution::seq;

#endif


    /// <summary>
    /// Constant used to represent a value that has not been computed.
    /// </summary>
    template<typename T>
    constexpr T NaN() { return std::numeric_limits<T>::quiet_NaN(); }

    /// <summary>
    /// Gets whether a value is computed or not.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    template<typename T>
    bool IsComputed(T value) { return !std::isnan(value); }

    // from https://stackoverflow.com/a/52886236/1272096
    namespace format_helper
    {
        template <typename T>
        inline T cast(T v) noexcept
        {
            return v;
        }

        inline const char* cast(const std::string& v) noexcept
        {
            return v.c_str();
        }
    }
        
    template <typename... Args>
    size_t FormatBuffer(
        char* const buffer, 
        size_t const bufferCount,
        char const* const format, 
        Args&&... args)
    {
        using namespace format_helper;
        
        auto const result = std::snprintf(buffer, bufferCount, format, cast(std::forward<Args>(args))...);
        assert(result != -1);
        
        return static_cast<size_t>(result);
    }

    template <typename... Args>
    inline void Format(
        std::string& buffer, 
        char const* const format,
        Args const&... args)
    {
        // from https://docs.microsoft.com/en-us/archive/msdn-magazine/2015/march/windows-with-c-using-printf-with-modern-c

        auto const size = FormatBuffer(&buffer[0], buffer.size() + 1, format, args...);
        if (size > buffer.size())
        {
            buffer.resize(size);
            FormatBuffer(&buffer[0], buffer.size() + 1, format, args...);
        }
        else if (size < buffer.size())
        {
            buffer.resize(size);
        }
    }

    template <typename... Args>
    inline std::string Format(
        char const* const format,
        Args&&... args)
    {
        std::string result;
        Format(result, format, std::forward<Args>(args)...);
        return result;
    }
}

