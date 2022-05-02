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

#include <version>

#if !defined(__cplusplus) || (__cplusplus < 201703L)
#error C++17 or higher is required; if compiling with msvc, please make sure to compile with /Zc:__cplusplus
#endif

#if defined(__GNUG__)
#include <tbb/tbb.h>
#endif

// uncomment the following to force sequential operation of parallel STL algorithms
// #define ALN_DEBUG_FORCE_SEQUENTIAL

#include "Helpers.h"
#include "Iterators.h"
#include "DataSet.h"
#include "DimensionalConstraint.h"
#include "FixedVector.h"
#include "NodeIndex.h"
#include "LinearUnit.h"
#include "MinMaxNode.h"
#include "Network.h"
#include "EvaluationResult.h"
#include "EvaluationRoute.h"
#include "Evaluate.h"
#include "Adapt.h"
#include "Split.h"
#include "Train.h"
#include "Json.h"
