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

#include <vector>

namespace aln
{
    /// <summary>
    /// A FixedVector{T} is like std::vector{T} except it cannot be dynamically resized.
    /// </summary>
    /// <remarks>
    /// Instance members are not safe for use across multiple threads without 
    /// external synchronization.
    /// </remarks>
    template <class T, class A = std::allocator<T>>
    struct FixedVector : private std::vector<T, A>
    {
        using FixedVector::vector::vector;
        using FixedVector::vector::operator=;
        using FixedVector::vector::get_allocator;
        using FixedVector::vector::at;
        using FixedVector::vector::front;
        using FixedVector::vector::back;
        using FixedVector::vector::data;
        using FixedVector::vector::begin;
        using FixedVector::vector::cbegin;
        using FixedVector::vector::end;
        using FixedVector::vector::cend;
        using FixedVector::vector::empty;
        using FixedVector::vector::size;
        using FixedVector::vector::operator[];
    };
}