//
//  solution_cycle.hpp
//  FXTrue
//
//  Created by Alexey Akhunov on 28/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#ifndef solution_cycle_h
#define solution_cycle_h

cl_uint solution_cycle_v2(cl::CommandQueue& queue,
                          cl::Buffer& cl_hash_chunks_src,
                          cl::Buffer& cl_sieve,
                          cl::Buffer& cl_pair_counts,
                          cl::Buffer& cl_partial_one,
                          cl::Buffer& cl_partial_two,
                          cl::Buffer& cl_pairs,
                          cl::Buffer& cl_hash_chunks_dst,
                          cl::Buffer& cl_schedule_v2,
                          cl::Buffer& cl_group_counts,
                          cl::Buffer& cl_group_offsets,
                          uint32_t* group_offsets,
                          cl::Kernel& clear_sieve_kernel,
                          std::vector<cl::Event>& clear_sieve_events,
                          cl::Kernel& fill_sieve_kernel,
                          std::vector<cl::Event>& fill_sieve_events,
                          cl::Kernel& correct_sieve_kernel,
                          std::vector<cl::Event>& correct_sieve_events,
                          cl::Kernel& align_collisions_kernel,
                          std::vector<cl::Event>& align_events,
                          cl::Kernel& group_count_kernel,
                          std::vector<cl::Event>& group_count_events,
                          cl::Kernel& scan_subarrays_kernel,
                          std::vector<cl::Event>& subarray_events,
                          cl::Kernel& scan_inc_subarrays_kernel,
                          std::vector<cl::Event>& subarray_inc_events,
                          cl::Kernel& group_offsets_kernel,
                          std::vector<cl::Event>& group_offsets_events,
                          cl::Kernel& clear_schedule_v2,
                          std::vector<cl::Event>& clear_schedule_v2_events,
                          cl::Kernel& project_v2,
                          std::vector<cl::Event>& project_v2_events,
                          cl::Kernel& group2_kernel,
                          std::vector<cl::Event>& group2_events,
                          cl::Kernel& group3_kernel,
                          std::vector<cl::Event>& group3_events,
                          cl::Kernel& group4_kernel,
                          std::vector<cl::Event>& group4_events,
                          cl::Kernel& group5_kernel,
                          std::vector<cl::Event>& group5_events,
                          cl::Kernel& groupc_kernel,
                          std::vector<cl::Event>& groupc_events,
                          const cl_uint KeysCount,
                          const cl_uint init_size,
                          const cl_uint max_workgroup_size,
                          const cl_uint chunks,
                          const cl_uint pairs_prev_offset,
                          const cl_uint pairs_offset,
                          const bool debug
                          )
{
    check(__LINE__,clear_sieve_kernel.setArg(0, cl_sieve));
    clear_sieve_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(clear_sieve_kernel, cl::NullRange, cl::NDRange{KeysCount*16}, cl::NullRange, nullptr, &clear_sieve_events.back()));
    check(__LINE__,fill_sieve_kernel.setArg(0, cl_hash_chunks_src));
    check(__LINE__,fill_sieve_kernel.setArg(1, cl_sieve));
    check(__LINE__,fill_sieve_kernel.setArg(2, (cl_uchar)(chunks+1)));
    fill_sieve_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(fill_sieve_kernel, cl::NullRange,cl::NDRange{init_size},cl::NullRange, nullptr, &fill_sieve_events.back()));
    check(__LINE__,correct_sieve_kernel.setArg(0, cl_hash_chunks_src));
    check(__LINE__,correct_sieve_kernel.setArg(1, cl_sieve));
    check(__LINE__,correct_sieve_kernel.setArg(2, (cl_uchar)(chunks+1)));
    correct_sieve_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(correct_sieve_kernel, cl::NullRange,cl::NDRange{init_size},cl::NullRange, nullptr, &correct_sieve_events.back()));
    check(__LINE__,align_collisions_kernel.setArg(0, cl_sieve));
    check(__LINE__,align_collisions_kernel.setArg(1, cl_pair_counts));
    align_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(align_collisions_kernel, cl::NullRange, cl::NDRange{16, KeysCount}, cl::NDRange{16, 16}, nullptr, &align_events.back()));
    check(__LINE__,group_count_kernel.setArg(0, cl_pair_counts));
    check(__LINE__,group_count_kernel.setArg(1, cl_group_counts));
    group_count_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group_count_kernel, cl::NullRange, cl::NDRange{KeysCount,15}, cl::NDRange{16,15}, nullptr, &group_count_events.back()));
    prefix_sum(queue, cl_group_counts, KeysCount*15+1, max_workgroup_size, cl_partial_one, cl_partial_two, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events);
    check(__LINE__,group_offsets_kernel.setArg(0, cl_group_counts));
    check(__LINE__,group_offsets_kernel.setArg(1, cl_group_offsets));
    check(__LINE__,group_offsets_kernel.setArg(2, KeysCount));
    group_offsets_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group_offsets_kernel, cl::NullRange, cl::NDRange{15}, cl::NullRange, nullptr, &group_offsets_events.back()));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_group_offsets, CL_TRUE, 0, sizeof(uint32_t)*15, group_offsets));
    check(__LINE__,clear_schedule_v2.setArg(0, cl_schedule_v2));
    clear_schedule_v2_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(clear_schedule_v2, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &clear_schedule_v2_events.back()));
    check(__LINE__,project_v2.setArg(0, cl_pair_counts));
    check(__LINE__,project_v2.setArg(1, cl_group_counts));
    check(__LINE__,project_v2.setArg(2, cl_schedule_v2));
    project_v2_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(project_v2, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &project_v2_events.back()));
    cl_uint dest_offset = 0;
    cl_uint size = group_offsets[0];
    cl_uint schedule_offset = 0;
    cl_uint num_groups = max_workgroup_size/chunks;
    check(__LINE__,group2_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group2_kernel.setArg(1, size));
    check(__LINE__,group2_kernel.setArg(2, cl_sieve));
    check(__LINE__,group2_kernel.setArg(3, cl_hash_chunks_src));
    check(__LINE__,group2_kernel.setArg(4, cl_hash_chunks_dst));
    check(__LINE__,group2_kernel.setArg(5, cl_pairs));
    check(__LINE__,group2_kernel.setArg(6, pairs_prev_offset));
    check(__LINE__,group2_kernel.setArg(7, pairs_offset));
    check(__LINE__,group2_kernel.setArg(8, chunks));
    check(__LINE__,group2_kernel.setArg(9, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,group2_kernel.setArg(10, cl::Local(sizeof(cl_uint)*2*num_groups)));
    group2_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group2_kernel, cl::NDRange{0}, cl::NDRange{chunks*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{num_groups*chunks}, nullptr, &group2_events.back()));
    dest_offset += 1*size;
    schedule_offset += size;
    size = group_offsets[1];
    num_groups = max_workgroup_size/chunks/3;
    check(__LINE__,group3_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group3_kernel.setArg(1, schedule_offset));
    check(__LINE__,group3_kernel.setArg(2, size));
    check(__LINE__,group3_kernel.setArg(3, cl_sieve));
    check(__LINE__,group3_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,group3_kernel.setArg(5, cl_hash_chunks_dst));
    check(__LINE__,group3_kernel.setArg(6, cl_pairs));
    check(__LINE__,group3_kernel.setArg(7, pairs_prev_offset));
    check(__LINE__,group3_kernel.setArg(8, pairs_offset));
    check(__LINE__,group3_kernel.setArg(9, dest_offset));
    check(__LINE__,group3_kernel.setArg(10, chunks));
    check(__LINE__,group3_kernel.setArg(11, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,group3_kernel.setArg(12, cl::Local(sizeof(cl_uint)*3*num_groups)));
    check(__LINE__,group3_kernel.setArg(13, cl::Local(sizeof(cl_uint)*3*num_groups*chunks)));
    group3_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group3_kernel, cl::NullRange, cl::NDRange{(3*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{3*num_groups*chunks}, nullptr, &group3_events.back()));
    dest_offset += 3*size;
    schedule_offset += size;
    size = group_offsets[2];
    num_groups = max_workgroup_size/chunks/4;
    check(__LINE__,group4_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group4_kernel.setArg(1, schedule_offset));
    check(__LINE__,group4_kernel.setArg(2, size));
    check(__LINE__,group4_kernel.setArg(3, cl_sieve));
    check(__LINE__,group4_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,group4_kernel.setArg(5, cl_hash_chunks_dst));
    check(__LINE__,group4_kernel.setArg(6, cl_pairs));
    check(__LINE__,group4_kernel.setArg(7, pairs_prev_offset));
    check(__LINE__,group4_kernel.setArg(8, pairs_offset));
    check(__LINE__,group4_kernel.setArg(9, dest_offset));
    check(__LINE__,group4_kernel.setArg(10, chunks));
    check(__LINE__,group4_kernel.setArg(11, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,group4_kernel.setArg(12, cl::Local(sizeof(cl_uint)*4*num_groups)));
    check(__LINE__,group4_kernel.setArg(13, cl::Local(sizeof(cl_uint)*4*num_groups*chunks)));
    group4_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group4_kernel, cl::NullRange, cl::NDRange{(4*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{4*num_groups*chunks}, nullptr, &group4_events.back()));
    dest_offset += 6*size;
    schedule_offset += size;
    size = group_offsets[3];
    num_groups = max_workgroup_size/chunks/5;
    check(__LINE__,group5_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group5_kernel.setArg(1, schedule_offset));
    check(__LINE__,group5_kernel.setArg(2, size));
    check(__LINE__,group5_kernel.setArg(3, cl_sieve));
    check(__LINE__,group5_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,group5_kernel.setArg(5, cl_hash_chunks_dst));
    check(__LINE__,group5_kernel.setArg(6, cl_pairs));
    check(__LINE__,group5_kernel.setArg(7, pairs_prev_offset));
    check(__LINE__,group5_kernel.setArg(8, pairs_offset));
    check(__LINE__,group5_kernel.setArg(9, dest_offset));
    check(__LINE__,group5_kernel.setArg(10, chunks));
    check(__LINE__,group5_kernel.setArg(11, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,group5_kernel.setArg(12, cl::Local(sizeof(cl_uint)*5*num_groups)));
    check(__LINE__,group5_kernel.setArg(13, cl::Local(sizeof(cl_uint)*5*num_groups*chunks)));
    group5_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group5_kernel, cl::NullRange, cl::NDRange{(5*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{5*num_groups*chunks}, nullptr, &group5_events.back()));
    dest_offset += 10*size;
    schedule_offset += size;
    check(__LINE__,groupc_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,groupc_kernel.setArg(3, cl_sieve));
    check(__LINE__,groupc_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,groupc_kernel.setArg(5, cl_hash_chunks_dst));
    check(__LINE__,groupc_kernel.setArg(6, cl_pairs));
    check(__LINE__,groupc_kernel.setArg(7, pairs_prev_offset));
    check(__LINE__,groupc_kernel.setArg(8, pairs_offset));
    check(__LINE__,groupc_kernel.setArg(10, (cl_uchar)chunks));
    
    for(uint count = 6;count<=16;++count) {
        size = group_offsets[count-2];
        if (size == 0) {
            continue;
        }
        num_groups = max_workgroup_size/chunks/count;
        check(__LINE__,groupc_kernel.setArg(1, schedule_offset));
        check(__LINE__,groupc_kernel.setArg(2, size));
        check(__LINE__,groupc_kernel.setArg(9, dest_offset));
        check(__LINE__,groupc_kernel.setArg(11, (cl_uchar)count));
        check(__LINE__,groupc_kernel.setArg(12, cl::Local(sizeof(cl_uint)*num_groups)));
        check(__LINE__,groupc_kernel.setArg(13, cl::Local(sizeof(cl_uint)*count*num_groups)));
        check(__LINE__,groupc_kernel.setArg(14, cl::Local(sizeof(cl_uint)*count*num_groups*chunks)));
        groupc_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(groupc_kernel, cl::NullRange, cl::NDRange{(count*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{count*num_groups*chunks}, nullptr, &groupc_events.back()));
        dest_offset += (count*(count-1)/2)*size;
        schedule_offset += size;
    }
    if (debug)
    {
        std::unique_ptr<uint32_t> hash_chunks_src(new uint32_t[10*init_size]);
        std::unique_ptr<uint32_t> hash_chunks_dst(new uint32_t[10*dest_offset]);
        std::unique_ptr<uint32_t> pairs(new uint32_t[2*dest_offset]);
        check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_dst, CL_TRUE, 0, sizeof(cl_uint)*10*dest_offset, hash_chunks_dst.get()));
        check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_src, CL_TRUE, 0, sizeof(cl_uint)*10*init_size, hash_chunks_src.get()));
        check(__LINE__,queue.enqueueReadBuffer(cl_pairs, CL_TRUE, sizeof(cl_uint)*pairs_offset, sizeof(cl_uint)*2*dest_offset, pairs.get()));
        size_t count_non_empty = 0;
        for(size_t i = 0;i<dest_offset;i++)
        {
            uint32_t i1 = pairs.get()[2*i];
            uint32_t i2 = pairs.get()[2*i+1];
            uint32_t chunk = hash_chunks_dst.get()[i*10];
            if (i1 != 0xFFFFFFFFU || i2 != 0xFFFFFFFFU) {
                assert(chunk == (hash_chunks_src.get()[10*i1+1]^hash_chunks_src.get()[10*i2+1]));
                for (int c = 1; c<chunks; c++)
                {
                    assert(hash_chunks_dst.get()[10*i+c] == (hash_chunks_src.get()[10*i1+c+1]^hash_chunks_src.get()[10*i2+c+1]));
                }
                count_non_empty++;
            }
        }
        std::cout << "Pairs in the hash_chunks_dst: " << count_non_empty << "\n";
    }
    return dest_offset;
}

cl_uint solution_cycle_final(cl::CommandQueue& queue,
                             cl::Buffer& cl_hash_chunks_src,
                             cl::Buffer& cl_sieve,
                             cl::Buffer& cl_pair_counts,
                             cl::Buffer& cl_partial_one,
                             cl::Buffer& cl_partial_two,
                             cl::Buffer& cl_pairs,
                             cl::Buffer& cl_schedule_v2,
                             cl::Buffer& cl_group_counts,
                             cl::Buffer& cl_group_offsets,
                             uint32_t* group_offsets,
                             cl::Kernel& clear_sieve_kernel,
                             std::vector<cl::Event>& clear_sieve_events,
                             cl::Kernel& fill_sieve_kernel,
                             std::vector<cl::Event>& fill_sieve_events,
                             cl::Kernel& correct_sieve_kernel,
                             std::vector<cl::Event>& correct_sieve_events,
                             cl::Kernel& align_collisions_kernel,
                             std::vector<cl::Event>& align_events,
                             cl::Kernel& group_count_kernel,
                             std::vector<cl::Event>& group_count_events,
                             cl::Kernel& scan_subarrays_kernel,
                             std::vector<cl::Event>& subarray_events,
                             cl::Kernel& scan_inc_subarrays_kernel,
                             std::vector<cl::Event>& subarray_inc_events,
                             cl::Kernel& group_offsets_kernel,
                             std::vector<cl::Event>& group_offsets_events,
                             cl::Kernel& clear_schedule_v2,
                             std::vector<cl::Event>& clear_schedule_v2_events,
                             cl::Kernel& project_v2,
                             std::vector<cl::Event>& project_v2_events,
                             cl::Kernel& final2_kernel,
                             std::vector<cl::Event>& group2_events,
                             cl::Kernel& final3_kernel,
                             std::vector<cl::Event>& group3_events,
                             cl::Kernel& final4_kernel,
                             std::vector<cl::Event>& group4_events,
                             cl::Kernel& final5_kernel,
                             std::vector<cl::Event>& group5_events,
                             cl::Kernel& finalc_kernel,
                             std::vector<cl::Event>& groupc_events,
                             const cl_uint KeysCount,
                             const cl_uint init_size,
                             const cl_uint max_workgroup_size,
                             const cl_uint pairs_prev_offset,
                             const cl_uint pairs_offset
                             )
{
    check(__LINE__,clear_sieve_kernel.setArg(0, cl_sieve));
    clear_sieve_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(clear_sieve_kernel, cl::NullRange, cl::NDRange{KeysCount*16}, cl::NullRange, nullptr, &clear_sieve_events.back()));
    check(__LINE__,fill_sieve_kernel.setArg(0, cl_hash_chunks_src));
    check(__LINE__,fill_sieve_kernel.setArg(1, cl_sieve));
    check(__LINE__,fill_sieve_kernel.setArg(2, (cl_uchar)2));
    fill_sieve_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(fill_sieve_kernel, cl::NullRange,cl::NDRange{init_size},cl::NullRange, nullptr, &fill_sieve_events.back()));
    check(__LINE__,correct_sieve_kernel.setArg(0, cl_hash_chunks_src));
    check(__LINE__,correct_sieve_kernel.setArg(1, cl_sieve));
    check(__LINE__,correct_sieve_kernel.setArg(2, (cl_uchar)2));
    correct_sieve_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(correct_sieve_kernel, cl::NullRange,cl::NDRange{init_size},cl::NullRange, nullptr, &correct_sieve_events.back()));
    check(__LINE__,align_collisions_kernel.setArg(0, cl_sieve));
    check(__LINE__,align_collisions_kernel.setArg(1, cl_pair_counts));
    align_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(align_collisions_kernel, cl::NullRange, cl::NDRange{16, KeysCount}, cl::NDRange{16, 16}, nullptr, &align_events.back()));
    check(__LINE__,group_count_kernel.setArg(0, cl_pair_counts));
    check(__LINE__,group_count_kernel.setArg(1, cl_group_counts));
    group_count_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group_count_kernel, cl::NullRange, cl::NDRange{KeysCount,15}, cl::NDRange{16,15}, nullptr, &group_count_events.back()));
    prefix_sum(queue, cl_group_counts, KeysCount*15+1, max_workgroup_size, cl_partial_one, cl_partial_two, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events);
    check(__LINE__,group_offsets_kernel.setArg(0, cl_group_counts));
    check(__LINE__,group_offsets_kernel.setArg(1, cl_group_offsets));
    check(__LINE__,group_offsets_kernel.setArg(2, KeysCount));
    group_offsets_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(group_offsets_kernel, cl::NullRange, cl::NDRange{15}, cl::NullRange, nullptr, &group_offsets_events.back()));
    check(__LINE__,queue.enqueueReadBuffer(cl_group_offsets, CL_TRUE, 0, sizeof(uint32_t)*15, group_offsets));
    check(__LINE__,clear_schedule_v2.setArg(0, cl_schedule_v2));
    clear_schedule_v2_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(clear_schedule_v2, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &clear_schedule_v2_events.back()));
    check(__LINE__,project_v2.setArg(0, cl_pair_counts));
    check(__LINE__,project_v2.setArg(1, cl_group_counts));
    check(__LINE__,project_v2.setArg(2, cl_schedule_v2));
    project_v2_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(project_v2, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &project_v2_events.back()));
    cl_uint dest_offset = 0;
    cl_uint size = group_offsets[0];
    cl_uint schedule_offset = 0;
    cl_uint num_groups = max_workgroup_size;
    check(__LINE__,final2_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,final2_kernel.setArg(1, size));
    check(__LINE__,final2_kernel.setArg(2, cl_sieve));
    check(__LINE__,final2_kernel.setArg(3, cl_hash_chunks_src));
    check(__LINE__,final2_kernel.setArg(4, cl_pairs));
    check(__LINE__,final2_kernel.setArg(5, pairs_prev_offset));
    check(__LINE__,final2_kernel.setArg(6, pairs_offset));
    check(__LINE__,final2_kernel.setArg(7, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,final2_kernel.setArg(8, cl::Local(sizeof(cl_uint)*2*num_groups)));
    group2_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(final2_kernel, cl::NDRange{0}, cl::NDRange{num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{num_groups}, nullptr, &group2_events.back()));
    dest_offset += 1*size;
    schedule_offset += size;
    size = group_offsets[1];
    num_groups = max_workgroup_size/3;
    check(__LINE__,final3_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,final3_kernel.setArg(1, schedule_offset));
    check(__LINE__,final3_kernel.setArg(2, size));
    check(__LINE__,final3_kernel.setArg(3, cl_sieve));
    check(__LINE__,final3_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,final3_kernel.setArg(5, cl_pairs));
    check(__LINE__,final3_kernel.setArg(6, pairs_prev_offset));
    check(__LINE__,final3_kernel.setArg(7, pairs_offset));
    check(__LINE__,final3_kernel.setArg(8, dest_offset));
    check(__LINE__,final3_kernel.setArg(9, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,final3_kernel.setArg(10, cl::Local(sizeof(cl_uint)*3*num_groups)));
    check(__LINE__,final3_kernel.setArg(11, cl::Local(sizeof(cl_uint)*3*num_groups)));
    group3_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(final3_kernel, cl::NullRange, cl::NDRange{3*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{3*num_groups}, nullptr, &group3_events.back()));
    dest_offset += 3*size;
    schedule_offset += size;
    size = group_offsets[2];
    num_groups = max_workgroup_size/4;
    check(__LINE__,final4_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,final4_kernel.setArg(1, schedule_offset));
    check(__LINE__,final4_kernel.setArg(2, size));
    check(__LINE__,final4_kernel.setArg(3, cl_sieve));
    check(__LINE__,final4_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,final4_kernel.setArg(5, cl_pairs));
    check(__LINE__,final4_kernel.setArg(6, pairs_prev_offset));
    check(__LINE__,final4_kernel.setArg(7, pairs_offset));
    check(__LINE__,final4_kernel.setArg(8, dest_offset));
    check(__LINE__,final4_kernel.setArg(9, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,final4_kernel.setArg(10, cl::Local(sizeof(cl_uint)*4*num_groups)));
    check(__LINE__,final4_kernel.setArg(11, cl::Local(sizeof(cl_uint)*4*num_groups)));
    group4_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(final4_kernel, cl::NullRange, cl::NDRange{4*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{4*num_groups}, nullptr, &group4_events.back()));
    dest_offset += 6*size;
    schedule_offset += size;
    size = group_offsets[3];
    num_groups = max_workgroup_size/5;
    check(__LINE__,final5_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,final5_kernel.setArg(1, schedule_offset));
    check(__LINE__,final5_kernel.setArg(2, size));
    check(__LINE__,final5_kernel.setArg(3, cl_sieve));
    check(__LINE__,final5_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,final5_kernel.setArg(5, cl_pairs));
    check(__LINE__,final5_kernel.setArg(6, pairs_prev_offset));
    check(__LINE__,final5_kernel.setArg(7, pairs_offset));
    check(__LINE__,final5_kernel.setArg(8, dest_offset));
    check(__LINE__,final5_kernel.setArg(9, cl::Local(sizeof(cl_uint)*num_groups)));
    check(__LINE__,final5_kernel.setArg(10, cl::Local(sizeof(cl_uint)*5*num_groups)));
    check(__LINE__,final5_kernel.setArg(11, cl::Local(sizeof(cl_uint)*5*num_groups)));
    group5_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(final5_kernel, cl::NullRange, cl::NDRange{5*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{5*num_groups}, nullptr, &group5_events.back()));
    dest_offset += 10*size;
    schedule_offset += size;
    check(__LINE__,finalc_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,finalc_kernel.setArg(3, cl_sieve));
    check(__LINE__,finalc_kernel.setArg(4, cl_hash_chunks_src));
    check(__LINE__,finalc_kernel.setArg(5, cl_pairs));
    check(__LINE__,finalc_kernel.setArg(6, pairs_prev_offset));
    check(__LINE__,finalc_kernel.setArg(7, pairs_offset));
    
    for(uint count = 6;count<=16;++count) {
        size = group_offsets[count-2];
        if (size == 0) {
            continue;
        }
        num_groups = max_workgroup_size/count;
        check(__LINE__,finalc_kernel.setArg(1, schedule_offset));
        check(__LINE__,finalc_kernel.setArg(2, size));
        check(__LINE__,finalc_kernel.setArg(8, dest_offset));
        check(__LINE__,finalc_kernel.setArg(9, (cl_uchar)count));
        check(__LINE__,finalc_kernel.setArg(10, cl::Local(sizeof(cl_uint)*num_groups)));
        check(__LINE__,finalc_kernel.setArg(11, cl::Local(sizeof(cl_uint)*count*num_groups)));
        check(__LINE__,finalc_kernel.setArg(12, cl::Local(sizeof(cl_uint)*count*num_groups)));
        groupc_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(finalc_kernel, cl::NullRange, cl::NDRange{count*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{count*num_groups}, nullptr, &groupc_events.back()));
        dest_offset += (count*(count-1)/2)*size;
        schedule_offset += size;
    }
    return dest_offset;
}

#endif /* solution_cycle_h */
