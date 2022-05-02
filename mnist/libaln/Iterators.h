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

#include "aln.h"

namespace aln
{
    /// <summary>
    /// Defines a range of values which may be iterated over.
    /// </summary>
    template <typename T>
    class Range
    {
    public:

        static_assert(std::is_arithmetic_v<T>);

        /// <summary>
        /// Constructs a new range.
        /// </summary>
        /// <param name="from">The beginning of the range.</param>
        /// <param name="to">One past the end of the range</param>
        Range(T from, T to) noexcept
            : _from(from), _to(to)
        {
            assert(from <= to);
        }

        class iterator
        {
        public:

            // iterator traits
            using value_type = T;
            using pointer = value_type const*;
            using reference = value_type const&;
            using difference_type = decltype(std::declval<value_type>() - std::declval<value_type>());
            using iterator_category = std::random_access_iterator_tag;

            iterator(value_type value = {}) noexcept
                : _value{ value }
            {
            }

            iterator& operator++() noexcept { _value += 1; return *this; }
            iterator operator++(int) noexcept { auto result = *this; ++(*this); return result; }

            iterator& operator+=(difference_type increment) noexcept { _value += increment; return *this; }
            iterator operator+(difference_type increment) const noexcept { auto result = *this; result += increment;  return result; }
            friend iterator operator+(difference_type increment, iterator const& it) noexcept { return it + increment; }

            iterator& operator--() noexcept { _value -= 1; return *this; }
            iterator operator--(int) noexcept { auto result = *this; --(*this); return result; }

            iterator& operator-=(difference_type decrement) noexcept { _value -= decrement; return *this; }
            iterator operator-(difference_type decrement) const noexcept { auto result = *this; result -= decrement;  return result; }

            difference_type operator-(iterator other) const noexcept { return _value - other._value; }

            bool operator==(iterator other) const noexcept { return _value == other._value; }
            bool operator!=(iterator other) const noexcept { return !(*this == other); }

            bool operator<(iterator other) const noexcept { return _value < other._value; }
            bool operator<=(iterator other) const noexcept { return _value <= other.row; }
            bool operator>(iterator other) const noexcept { return _value > other._value; }
            bool operator>=(iterator other) const noexcept { return _value >= other._value; }
            
            value_type operator[](difference_type index) const noexcept { return _value + index; }

            value_type operator*() const noexcept { return _value; }
            pointer operator->() const noexcept { return &_value; }

        private:
            friend class Range;
            value_type _value;
        };


        iterator begin() { return iterator(_from); }
        iterator end() { return iterator(_to); }

        iterator begin() const { return iterator(_from); }
        iterator end() const { return iterator(_to); }

        iterator cbegin() const { return begin(); }
        iterator cend() const { return end(); }

        T size() const { return _to - _from; }

    private:

        T _from;
        T _to;
    };

    /// <summary>
    /// Makes a range from the specified values.
    /// </summary>
    template <typename T>
    Range<T> make_range(T from, T to)
    {
        return Range<T>(from, to);
    }

    /// <summary>
    /// Makes a range from a default initialized value of {T} to the specified end value.
    /// </summary>
    template <typename T>
    Range<T> make_range(T to)
    {
        return Range<T>(0, to);
    }

    template <class T>
    [[nodiscard]] constexpr auto begin(Range<T>& range) noexcept { return range.begin(); }

    template <class T>
    [[nodiscard]] constexpr auto begin(Range<T> const& range) noexcept { return range.begin(); }

    template <class T>
    [[nodiscard]] constexpr auto end(Range<T>& range) noexcept { return range.end(); }

    template <class T>
    [[nodiscard]] constexpr auto end(Range<T> const& range) noexcept { return range.end(); }

    template <class T>
    [[nodiscard]] constexpr auto cbegin(const Range<T>& range) noexcept { return begin(range); }

    template <class T>
    [[nodiscard]] constexpr auto cend(const Range<T>& range) noexcept { return end(range); }


    /// <summary>
    /// Defines a data row iterator.  Iterates through elements of a data row. No bounds checking done.
    /// </summary>
    template <typename T>
    class RowElementIterator
    {
    public:

        // iterator traits
        using value_type = T;
        using pointer = value_type const*;
        using reference = value_type const&;
        using difference_type = ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        RowElementIterator(T const* const firstElement = nullptr) noexcept
            : _element{ firstElement }
        {
        }

        RowElementIterator& operator++() noexcept { _element += 1; return *this; }
        RowElementIterator operator++(int) noexcept { auto result = *this; ++(*this); return result; }

        RowElementIterator& operator+=(difference_type increment) noexcept { _element += increment; return *this; }
        RowElementIterator operator+(difference_type increment) const noexcept { auto result = *this; result += increment;  return result; }
        friend RowElementIterator operator+(difference_type increment, RowElementIterator const& it) noexcept { return it + increment; }

        RowElementIterator& operator--() noexcept { _element -= 1; return *this; }
        RowElementIterator operator--(int) noexcept { auto result = *this; --(*this); return result; }

        RowElementIterator& operator-=(difference_type decrement) noexcept { _element -= decrement; return *this; }
        RowElementIterator operator-(difference_type decrement) const noexcept { auto result = *this; result -= decrement;  return result; }

        difference_type operator-(RowElementIterator other) const noexcept { return _element - other._element; }

        bool operator==(RowElementIterator other) const noexcept { return _element == other._element; }
        bool operator!=(RowElementIterator other) const noexcept { return _element != other._element; }

        bool operator<(RowElementIterator other) const noexcept { return _element < other._element; }
        bool operator<=(RowElementIterator other) const noexcept { return _element <= other.row; }
        bool operator>(RowElementIterator other) const noexcept { return _element > other._element; }
        bool operator>=(RowElementIterator other) const noexcept { return _element >= other._element; }

        reference operator[](size_t index) const noexcept { return _element[index]; }

        reference operator*() const noexcept { return *_element; }
        pointer operator->() const noexcept { return _element; }

    private:

        value_type const* _element;
    };

    /// <summary>
    /// Defines a row-major data matrix iterator.  Iterates through rows of a data matrix.  No bounds checking done.
    /// </summary>
    template <typename T>
    class RowMajorMatrixIterator
    {
    public:

        // iterator traits
        using value_type = RowElementIterator<T>;
        using pointer = RowElementIterator<T> const*;
        using reference = RowElementIterator<T> const&;
        using difference_type = ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        RowMajorMatrixIterator(T const* const firstRow, size_t columns) noexcept
            : _row{ firstRow }
            , _stride{ columns }
        {
            assert(columns > 0);
        }

        RowMajorMatrixIterator& operator++() noexcept { _row += _stride; return *this; }
        RowMajorMatrixIterator operator++(int) noexcept { auto result = *this; ++(*this); return result; }

        RowMajorMatrixIterator& operator+=(int increment) noexcept { _row += _stride * increment; return *this; }
        RowMajorMatrixIterator operator+(int increment) const noexcept { auto result = *this; result += increment;  return result; }
        friend RowMajorMatrixIterator operator+(difference_type increment, RowMajorMatrixIterator const& it) noexcept { return it + increment; }

        RowMajorMatrixIterator& operator--() noexcept { _row -= _stride; return *this; }
        RowMajorMatrixIterator operator--(int) noexcept { auto result = *this; --(*this); return result; }

        RowMajorMatrixIterator& operator-=(int decrement) noexcept { _row -= _stride * decrement; return *this; }
        RowMajorMatrixIterator operator-(int decrement) const noexcept { auto result = *this; result -= decrement;  return result; }

        difference_type operator-(RowMajorMatrixIterator other) const noexcept { return (_row - other._row) / _stride; }

        bool operator==(RowMajorMatrixIterator other) const noexcept { return _row == other._row; }
        bool operator!=(RowMajorMatrixIterator other) const noexcept { return !(*this == other); }

        bool operator<(RowMajorMatrixIterator other) const noexcept { return _row < other._row; }
        bool operator<=(RowMajorMatrixIterator other) const noexcept { return _row <= other.row; }
        bool operator>(RowMajorMatrixIterator other) const noexcept { return _row > other._row; }
        bool operator>=(RowMajorMatrixIterator other) const noexcept { return _row >= other._row; }

        value_type operator[](size_t index) const noexcept { return _row + (index * _stride); }

        reference operator*() const noexcept { return *_row; }
        pointer operator->() const noexcept { return _row; }

    private:

        RowElementIterator<T> _row;
        size_t _stride;
    };


    template <class T>
    [[nodiscard]] constexpr auto begin(RowElementIterator<T>& point) noexcept { return &(point[0]); }

    template <class T>
    [[nodiscard]] constexpr auto begin(RowElementIterator<T> const& point) noexcept { return &(point[0]); }

    template <class T>
    [[nodiscard]] constexpr auto cbegin(const RowElementIterator<T>& point) noexcept { return &(point[0]); }

    template <class T>
    [[nodiscard]] constexpr auto begin(RowMajorMatrixIterator<T>& dataSet) noexcept { return dataSet[0]; }

    template <class T>
    [[nodiscard]] constexpr auto begin(RowMajorMatrixIterator<T> const& dataSet) noexcept { return dataSet[0]; }

    template <class T>
    [[nodiscard]] constexpr auto cbegin(const RowMajorMatrixIterator<T>& dataSet) noexcept { return dataSet[0]; }

}