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

#include <optional>
#include <type_traits>

#include "aln.h"

namespace aln
{
    /// <summary>
    /// Represents the structure of an Adaptive Logic Network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instance members are not safe for use across multiple threads without 
    /// external synchronization.
    /// </para>
    /// <para>
    /// Many methods may internally invoke operations that make use
    /// of parallel execution capabilities of the host, but all such operations will be 
    /// completed synchronously before these methods return to the caller.
    /// </para>
    /// </remarks>
    template<typename T>
    class Network final
    {
    public:

        static_assert(std::is_floating_point_v<T>);

        /// <summary>
        /// Constructs a new instance of <see cref="Network"/> suitable for deserialization via move assignment.
        /// </summary>
        Network()
            : _dimensions{ }
        {
        }

        /// <summary>
        /// Constructs a new instance of <see cref="Network"/>.
        /// </summary>
        /// <param name="dimensions">The number of independent variables in the domain of the function computed by the network.</param>
        /// <param name="isSplittingAllowed"></param>
        Network(size_t dimensions, bool isSplittingAllowed = true)
            : _dimensions{ dimensions }
            , _constraints(dimensions)
        {
            if (dimensions < 1)
                throw std::logic_error("invalid dimensions");

            auto& luRoot = _linearUnits.emplace_back(
                _dimensions, 
                NodeIndex::Invalid());

            _rootIndex = NodeIndex::Linear(0);

            luRoot.SetIsSplitAllowed(isSplittingAllowed);

            // init lu average distances from centroid (D); they cannot be zero
            auto const range = make_range(_dimensions);
            auto& d = luRoot.D();
            std::for_each_n(
                Sequential,
                cbegin(range),
                _dimensions,
                [&](auto i)
                {
                    auto const& constraint = _constraints[i];
                    d[i] = constraint.Epsilon() * constraint.Epsilon();
                    assert(d[i] > 0);
                });
        }

        /// <summary>
        /// Constructs a new instance of <see cref="Network"/> from existing values.
        /// </summary>
        /// <param name="dimensions">The number of independent variables in the domain of the function computed by the network.</param>
        /// <param name="rootIndex"></param>
        /// <param name="minMaxNodes"></param>
        /// <param name="linearUnits"></param>
        /// <remarks>No validation is currently done on the values.  The caller should ensure that everything is consistent.</remarks>
        Network(size_t dimensions, 
            NodeIndex rootIndex,
            std::vector<MinMaxNode<T>> minMaxNodes,
            std::vector<LinearUnit<T>> linearUnits)
            : _dimensions{ dimensions }
            , _constraints(dimensions)
            , _rootIndex{ rootIndex }
            , _minMaxNodes { std::move(minMaxNodes) }
            , _linearUnits { std::move(linearUnits) }
        {
            if (dimensions < 1)
                throw std::logic_error("invalid dimensions");

            // todo: validate
        }

        /// <summary>
        /// Gets whether the specified index is valid for the network.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        bool IsValidIndex(NodeIndex index) const
        {
            if (index.IsInvalid()) return false;

            return index.IsLinear()
                ? index.Ordinal() < _linearUnits.size()
                : index.Ordinal() < _minMaxNodes.size();
        }

        /// <summary>
        /// Gets the index of the specified <paramref name="minMaxNode"/>.
        /// </summary>
        /// <param name="minMaxNode"></param>
        /// <returns>the index of the specified <paramref name="minMaxNode"/>, or <see cref="NodeIndex.Invalid()"/> if the node is not contained in the network.</returns>
        NodeIndex IndexOf(MinMaxNode<T> const& minMaxNode) const
        {
            auto address = &minMaxNode;
            auto front = &_minMaxNodes.front();
            if (address < front || address > & _minMaxNodes.back())
                return NodeIndex::Invalid();

            auto const ordinal = address - front;
            return NodeIndex::MinMax(ordinal);
        }

        /// <summary>
        /// Gets the index of the specified <paramref name="linearUnit"/>.
        /// </summary>
        /// <param name="linearUnit"></param>
        /// <returns>the index of the specified <paramref name="linearUnit"/>, or <see cref="NodeIndex.Invalid()"/> if the unit is not contained in the network.</returns>
        NodeIndex IndexOf(LinearUnit<T> const& linearUnit) const
        {
            auto address = &linearUnit;
            auto front = &_linearUnits.front();
            if (address < front || address > & _linearUnits.back())
                return NodeIndex::Invalid();

            auto const ordinal = address - front;
            return NodeIndex::Linear(ordinal);
        }

        /// <summary>
        /// Gets the <see cref="MinMaxNode"/> at the specified <paramref name="index"/>.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        MinMaxNode<T>& MinMaxNodeAt(NodeIndex index)
        {
            if (!index.IsMinMax())
                throw std::invalid_argument("invalid min/max node index");

            return _minMaxNodes[index.Ordinal()];
        }

        /// <summary>
        /// Gets the <see cref="MinMaxNode"/> at the specified <paramref name="index"/>.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        MinMaxNode<T> const& MinMaxNodeAt(NodeIndex index) const
        {
            if (!index.IsMinMax())
                throw std::invalid_argument("invalid min/max node index");

            return _minMaxNodes[index.Ordinal()];
        }

        /// <summary>
        /// Gets the <see cref="MinMaxNode"/> at the specified <paramref name="ordinal"/>.
        /// </summary>
        /// <param name="ordinal"></param>
        /// <returns></returns>
        MinMaxNode<T>& MinMaxNodeAt(size_t ordinal)
        {
            return _minMaxNodes[ordinal];
        }

        /// <summary>
        /// Gets the <see cref="MinMaxNode"/> at the specified <paramref name="ordinal"/>.
        /// </summary>
        /// <param name="ordinal"></param>
        /// <returns></returns>
        MinMaxNode<T> const& MinMaxNodeAt(size_t ordinal) const
        {
            return _minMaxNodes[ordinal];
        }

        /// <summary>
        /// Gets the vector of min / max nodes in the network.
        /// </summary>
        /// <returns></returns>
        std::vector<MinMaxNode<T>> const& MinMaxNodes() const { return _minMaxNodes; }


        /// <summary>
        /// Gets the <see cref="LinearUnit"/> at the specified <paramref name="ordinal"/>.
        /// </summary>
        /// <param name="ordinal"></param>
        /// <returns></returns>
        LinearUnit<T>& LinearUnitAt(size_t ordinal)
        {
            return _linearUnits[ordinal];
        }

        /// <summary>
        /// Gets the <see cref="LinearUnit"/> at the specified <paramref name="ordinal"/>.
        /// </summary>
        /// <param name="ordinal"></param>
        /// <returns></returns>
        LinearUnit<T> const& LinearUnitAt(size_t ordinal) const
        {
            return _linearUnits[ordinal];
        }

        /// <summary>
        /// Gets the <see cref="LinearUnit"/> at the specified <paramref name="index"/>.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        LinearUnit<T>& LinearUnitAt(NodeIndex index)
        {
            if (!index.IsLinear())
                throw std::invalid_argument("invalid linear unit index");

            return _linearUnits[index.Ordinal()];
        }

        /// <summary>
        /// Gets the <see cref="LinearUnit"/> at the specified <paramref name="index"/>.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        LinearUnit<T> const& LinearUnitAt(NodeIndex index) const
        {
            if (!index.IsLinear())
                throw std::invalid_argument("invalid linear unit index");

            return _linearUnits[index.Ordinal()];
        }

        /// <summary>
        /// Gets the vector of linear units in the network.
        /// </summary>
        /// <returns></returns>
        std::vector<LinearUnit<T>> const& LinearUnits() const { return _linearUnits; }

        /// <summary>
        /// Gets the root child index.
        /// </summary>
        /// <returns></returns>
        NodeIndex RootIndex() const
        {
            return _rootIndex;
        }

        /// <summary>
        /// Gets the number of independent variables in the domain of the function computed by the network.
        /// </summary>
        /// <returns></returns>
        size_t Dimensions() const
        {
            return _dimensions;
        }

        /// <summary>
        /// Gets the minimum allowable weight for a independent variable of the network.
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns>The minimum allowable weight for an independent variable of the network</returns>
        T MinWeight(size_t dimension) const
        {
            return _constraints[dimension].MinWeight();
        }

        /// <summary>
        /// Sets the minimum allowable weight for an independent variable of the network.
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="value"></param>
        void SetMinWeight(size_t dimension, T value)
        {
            if (_dimensions == 0)
                throw std::logic_error("zero dimensional network");

            _constraints[dimension].SetMinWeight(value);
            EnforceConstraints(dimension);
        }

        /// <summary>
        /// Gets the maximum allowable weight for an independent variable of the network.
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns>The maximum allowable weight for an independent variable of the network</returns>
        T MaxWeight(size_t dimension) const
        {
            return _constraints[dimension].MaxWeight();
        }

        /// <summary>
        /// Sets the maximum allowable weight for a independent variable of the network.
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="value"></param>
        void SetMaxWeight(size_t dimension, T value)
        {
            if (_dimensions == 0)
                throw std::logic_error("zero dimensional network");

            _constraints[dimension].SetMaxWeight(value);
            EnforceConstraints(dimension);
        }

        /// <summary>
        /// Gets the epsilon constraint for the specified dimension.
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        T Epsilon(size_t dimension) const
        {
            return _constraints[dimension].Epsilon();
        }

        /// <summary>
        /// Sets epsilon constraint for a dimension of the network.  This implicitly sets the
        /// value of <see cref="SquaredEpsilon()"/>.
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="value"></param>
        void SetEpsilon(size_t dimension, T value)
        {
            if (_dimensions == 0)
                throw std::logic_error("zero dimensional network");

            _constraints[dimension].SetEpsilon(value);
            EnforceConstraints(dimension);
        }

        /// <summary>
        /// Gets the vector of constraints on the network.
        /// </summary>
        /// <returns></returns>
        FixedVector<DimensionalConstraint<T>> const& Constraints() const 
        { 
            return _constraints; 
        }

        /// <summary>
        /// Splits a linear unit into the specified number of pieces, each of which
        /// will be a copy of the original linear unit with slight different weights.
        /// </summary>
        /// <param name="splitIndex">The index of the linear unit to split.</param>
        /// <param name="minOrMax">The type of min/max node to replace the linear unit with.</param>
        /// <param name="pieceCount">The number of linear pieces to split the original linear unit into.</param>
        /// <param name="equivalenceAdjustment">
        /// A non-negative value indicating the amount to adjust new pieces so they are not precisely equivalent
        /// to the original linear unit.
        /// </param>
        /// <returns>The index of the new min/max node in the network.</returns>
        /// <remarks>
        /// Because internal storage may be reallocated during splitting, previously held references to <see cref="MinMaxNode"/>
        /// or <see cref="LinearUnit"/> instances may no longer be valid.  Additionally, this operation may also invalidate
        /// the instance's <see cref="RootIndex()"/> if it was a <see cref="LinearNnit"/>.
        /// </remarks>
        NodeIndex Split(NodeIndex splitIndex, MinMaxType minOrMax, size_t pieceCount = 2, T equivalenceAdjustment = static_cast<T>(0.00001))
        {
            if (_dimensions == 0)
                throw std::logic_error("zero dimensional network");

            if (!splitIndex.IsLinear())
                throw std::invalid_argument("invalid split index");

            if (pieceCount < 2)
                throw std::invalid_argument("invalid piece count");

            if (equivalenceAdjustment < 0)
                throw std::invalid_argument("invalid equivalence adjustment");

            // adjust so the pieces aren't precisely equal (not currently used)
            auto const adjustment = minOrMax == MinMaxType::Min
                ? equivalenceAdjustment
                : -equivalenceAdjustment;

            auto nextMinMaxOrdinal = _minMaxNodes.size();
            NodeIndex result = NodeIndex::MinMax(nextMinMaxOrdinal);

            while (true)
            {
                // the linear unit to be split will become left child of the new min/max node;
                auto const leftIndex = NodeIndex::Linear(splitIndex.Ordinal());

                auto& leftLu = LinearUnitAt(leftIndex);
                if (!leftLu.IsSplitAllowed())
                    throw std::invalid_argument("split is not allowed for the specified linear unit");

                // save parent index
                auto parentIndex = leftLu.ParentIndex();

                // right child is a copy of left; 
                auto const rightOrdinal = _linearUnits.size();
                auto const rightIndex = NodeIndex::Linear(rightOrdinal);

                // note: leftLu reference may now be invalid if _linearUnits storage reallocated
                _linearUnits.push_back(leftLu);
                auto& rightLu = LinearUnitAt(rightIndex);

                // calculate index of next min/max node
                nextMinMaxOrdinal = _minMaxNodes.size();
                auto const minMaxIndex = NodeIndex::MinMax(nextMinMaxOrdinal);

                // update root or child index in parent
                if (parentIndex.IsInvalid())
                {
                    // this is the case when we're adapting the very first LTU
                    assert(_rootIndex.IsLinear());
                    assert(splitIndex.Ordinal() == 0); // must the first linear unit

                    // root is now a min max node
                    _rootIndex = minMaxIndex;
                }
                else
                {
                    // fixup child index in existing parent since we're changing type
                    auto& existingParent = MinMaxNodeAt(parentIndex);
                    existingParent.ReplaceChild(splitIndex, minMaxIndex);
                }

                // add min/max node
                auto& minMaxNode = _minMaxNodes.emplace_back(minOrMax, parentIndex, leftIndex, rightIndex);
                assert(IndexOf(minMaxNode) == minMaxIndex);

                // fixup parent index of children
                auto const& children = minMaxNode.Children();
                std::for_each_n(
                    begin(children),
                    2,
                    [&](NodeIndex childIndex)
                    {
                        if (childIndex.IsLinear())
                        {
                            LinearUnitAt(childIndex).ReplaceParent(minMaxIndex);
                        }
                        else
                        {
                            MinMaxNodeAt(childIndex).ReplaceParent(minMaxIndex);
                        }
                    });


                // are we finished?
                if (pieceCount == 2)
                    break;

                // ... no; split the new right child        
                pieceCount--;
                splitIndex = rightIndex;
            }

            return result;
        }


    private:

        /// <summary>
        /// Enforces constraints for the specified dimension.
        /// </summary>
        void EnforceConstraints(size_t dimension)
        {
            assert(dimension < _dimensions);

            auto const& constraint = _constraints[dimension];

            if (constraint.MaxWeight() < constraint.MinWeight())
                throw std::logic_error("constraint's maximum weight is less then it's minimum");

            if (constraint.Epsilon() <= 0)
                throw std::logic_error("constraint's epsilon must be positive");

            std::for_each_n(
                ParallelUnsequenced,
                begin(_linearUnits),
                _linearUnits.size(),
                [=](LinearUnit<T>& lu)
                {
                    if (lu.IsConstant())
                        return;

                    auto& w = lu.W();
                    auto& d = lu.D();

                    w[dimension] = std::clamp(w[dimension], constraint.MinWeight(), constraint.MaxWeight());
                    d[dimension] = std::max(d[dimension], constraint.Epsilon() * constraint.Epsilon());
                });
        }

        size_t                                _dimensions;  // number of dimensions in the domain of the function
        FixedVector<DimensionalConstraint<T>> _constraints; // constraints on each dimension
        NodeIndex                             _rootIndex;   // index of the root node in the network
        std::vector<MinMaxNode<T>>            _minMaxNodes;
        std::vector<LinearUnit<T>>            _linearUnits;
    };
}