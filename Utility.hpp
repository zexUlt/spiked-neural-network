#include "NumCpp.hpp"

#include <iostream>

class UtilityFunctionLibrary
{
public:
    template<typename dtype>
    static nc::NdArray<dtype> reverseRows(nc::NdArray<dtype>& a)
    {
        nc::NdArray<dtype> out(1, a.numCols());

        for(nc::uint32 i = 0; i < a.numRows(); ++i){
            out = nc::append(out, a(a.numRows() - 1u - i, a.cSlice()), nc::Axis::ROW);
        }


        return out(nc::Slice(1, out.numRows()), out.cSlice());
    }

    template<typename dtype>
    static nc::NdArray<dtype> convolveValid(const nc::NdArray<dtype>& f, const nc::NdArray<dtype>& g)
    {
        const auto nf = f.size();
        const auto ng = g.size();
        const auto& min_v = (nf < ng) ? f : g;
        const auto& max_v = (nf < ng) ? g : f;
        const auto n = std::max(nf, ng) - std::min(nf, ng) + 1;
        nc::NdArray<dtype> out(1, n);

        for(auto i(0u); i < n; ++i){
            for(int j(min_v.size() - 1), k(i); j >=0; --j, ++k){
                out.at(i) += min_v[j] * max_v[k];
            }
        }

        return out;
    }

    template< template<typename dtype> class T, typename dtype>
    static nc::DataCube<dtype> construct_fill_DC(const T<dtype>& init_val, nc::uint32 capacity)
    {
        nc::DataCube<dtype> out(capacity);

        for(int i = 0; i < capacity; ++i){
            out.push_back(init_val);
        }

        return out;
    }
};
