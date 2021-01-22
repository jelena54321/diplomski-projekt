#include "generate_features.h"
#include "models.h"

std::unordered_map<Bases, uint8_t, EnumClassHash> ENCODED_BASES = {
        {Bases::A, 0},
        {Bases::C, 1},
        {Bases::G, 2},
        {Bases::T, 3},
        {Bases::GAP, 4},
        {Bases::UNKNOWN, 5}
};

std::unique_ptr<Data> generate_features(const char *file_name, const char *ref, const char *region) {
    auto data = std::unique_ptr<Data>(new Data());

    std::vector<std::pair<long, long>> position_queue;
    std::unordered_map<std::pair<long, long>, std::unordered_map<uint32_t, PosInfo>, pair_hash> align_info;
    std::unordered_map<uint32_t, std::pair<long, long>> align_bounds;
    std::unordered_map<uint32_t, bool> strand;

    auto bam_file = openBAMFile(file_name);
    auto pileup_iter = bam_file->pileup(region);

    while (pileup_iter->has_next()) {
        auto column = pileup_iter->next();

        long ref_position = column->position;
        if (ref_position < pileup_iter->start()) continue;
        if (ref_position >= pileup_iter->end()) break;

        while(column->has_next()) {
            auto r = column->next();

            if (r->is_refskip()) continue;

            if (align_bounds.find(r->query_id()) == align_bounds.end()) {
                align_bounds.emplace(r->query_id(), std::make_pair(r->ref_start(), r->ref_end()));
            }
            strand.emplace(r->query_id(), !r->rev());

            std::pair<long, long> index(ref_position, 0);
            if (align_info.find(index) == align_info.end()) {
                position_queue.emplace_back(ref_position, 0);
            }

            if (r->is_del()) {
                // deletion
                align_info[index].emplace(r->query_id(), PosInfo(Bases::GAP));
            } else {
                auto query_base = r->qbase(0);
                align_info[index].emplace(r->query_id(), PosInfo(query_base));

                // insertion
                for (int i = 1, n = std::min(r->indel(), MAX_INS); i <= n; i++) {
                    index = std::pair<long, long>(ref_position, i);

                    if (align_info.find(index) == align_info.end()) {
                        position_queue.emplace_back(ref_position, i);
                    }

                    query_base = r->qbase(i);
                    align_info[index].emplace(r->query_id(), PosInfo(query_base));
                }
            }
        }

        // building a feature matrix
        while (position_queue.size() >= dimensions[1]) {

            const auto it = position_queue.begin();

            // remove aligns with an unknown base
            std::set<uint32_t> valid_aligns;
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s;
                for (auto& align : align_info[*curr]) {
                    if (align.second.base != Bases::UNKNOWN) {
                        valid_aligns.emplace(align.first);
                    }
                }
            }

            std::vector<uint32_t> valid(valid_aligns.begin(), valid_aligns.end());
            int valid_size = valid.size();

            // initialize feature matrix
            npy_intp dims[2];
            for (int i = 0; i < 2; i++) dims[i] = dimensions[i];
            auto X = PyArray_SimpleNew(2, dims, NPY_UINT8);

            uint8_t* value_ptr;

            // fill first REF_ROWS with ref
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s; 

                uint8_t value;
                if (curr->second != 0) value = ENCODED_BASES[Bases::GAP];
                else value = ENCODED_BASES[get_base(ref[curr->first])];

                for (int r = 0; r < REF_ROWS; r++) {
                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = value;
                }
            }

            // fill remaining (dimension[0] - REF_ROWS) rows with aligned reads
            for (int r = REF_ROWS; r < dimensions[0]; r++) {
                uint32_t query_id = valid[rand() % valid_size];

                // auto it = position_queue.begin();
                for (auto s = 0; s < dimensions[1]; s++) {
                    auto curr = it + s;

                    auto pos_itr = align_info[*curr].find(query_id);
                    auto& bounds = align_bounds[query_id];
                    uint8_t base;
                    if (pos_itr == align_info[*curr].end()) {
                        if (curr->first < bounds.first || curr->first > bounds.second) {
                            base = ENCODED_BASES[Bases::UNKNOWN];
                        } else {
                            base = ENCODED_BASES[Bases::GAP];
                        }
                    } else {
                        base = ENCODED_BASES[pos_itr->second.base];
                    }

                    auto& fwd = strand[query_id];
                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = fwd ? base : (base + 6);
                }
            }

            data->X.push_back(X);
            data->positions.emplace_back(position_queue.begin(), position_queue.begin() + dimensions[1]);

            for (auto it = position_queue.begin(), end = position_queue.begin() + WINDOW; it != end; it++) {
                align_info.erase(*it);
            }
            position_queue.erase(position_queue.begin(), position_queue.begin() + WINDOW);
        }
    }

    return data;
}
