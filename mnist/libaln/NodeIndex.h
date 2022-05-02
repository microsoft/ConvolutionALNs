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

namespace aln
{
    /// <summary>
    /// A NodeIndex is an unsigned integer that encodes a network nodes's type and ordinal position 
    /// within a sequence of nodes of the same type within a network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default constructed instances do not represent a valid ordinal position or type.
    /// </para>
    /// <para>
    /// By encoding the network node type in the index, we avoid polymorphic node types.
    /// </para>
    /// </remarks>
    struct NodeIndex final
    {
    public:

        constexpr NodeIndex()
            : NodeIndex(InvalidFlag)
        {
        }

        /// <summary>
        /// An invalid index value.
        /// </summary>
        /// <returns></returns>
        static NodeIndex Invalid()
        {
            return NodeIndex();
        }

        /// <summary>
        /// An index value for a linear unit child node.
        /// </summary>
        /// <returns></returns>
        static NodeIndex Linear(size_t ordinal)
        {
            return MakeLinearIndex(ordinal);
        }

        /// <summary>
        /// An index value for a min/max child or parent node.
        /// </summary>
        /// <returns></returns>
        static NodeIndex MinMax(size_t ordinal)
        {
            return MakeMinMaxIndex(ordinal);
        }

        constexpr bool   IsInvalid() const { return (_value & InvalidFlag) == InvalidFlag; }
        constexpr bool   IsLinear()  const { return (_value & LinearFlag)  == LinearFlag; }
        constexpr bool   IsMinMax()  const { return (_value & MinMaxFlag)  == MinMaxFlag; }
        constexpr size_t Ordinal()   const { assert(!IsInvalid()); return _value & OrdinalMask; }

        constexpr size_t Hash()      const { return _value; }

        constexpr bool operator==(NodeIndex const& other) const { return _value == other._value; }
        constexpr bool operator!=(NodeIndex const& other) const { return _value != other._value; }

    private:

        constexpr NodeIndex(size_t value)
            : _value{ value }
        {
        }

        constexpr static size_t MakeLinearIndex(size_t ordinal)
        {
            assert((ordinal & FlagMask) == 0);
            return ((ordinal & OrdinalMask) | LinearFlag);
        }

        constexpr static size_t MakeMinMaxIndex(size_t ordinal)
        {
            assert((ordinal & FlagMask) == 0);
            return ((ordinal & OrdinalMask) | MinMaxFlag);
        }

        static_assert(sizeof(size_t) >= 8);
        static constexpr size_t AvailableBits = sizeof(size_t) * 8;
        static constexpr size_t InvalidFlag   = static_cast<size_t>(1) << (AvailableBits - 1);
        static constexpr size_t LinearFlag    = static_cast<size_t>(1) << (AvailableBits - 2);
        static constexpr size_t MinMaxFlag    = static_cast<size_t>(1) << (AvailableBits - 3);
        static constexpr size_t FlagMask      = InvalidFlag | LinearFlag | MinMaxFlag;
        static constexpr size_t OrdinalMask   = ~FlagMask;

        size_t _value;
    };
}

namespace std 
{
    /// <summary>
    /// Implement std::hash{} specialization for <see cref="aln.NodeIndex"/>.
    /// </summary>
    template<>
    struct hash<aln::NodeIndex>
    {
        size_t operator()(aln::NodeIndex value) const
        {
            return value.Hash();
        }
    };
}
