//
//  opencl_solve.hpp
//  FXTrue
//
//  Created by Alexey Akhunov on 28/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#ifndef opencl_solve_h
#define opencl_solve_h

#if defined(__APPLE__) || defined(__MACOSX)
#include "cl.hpp"
#else
#include <CL/cl.hpp>
#endif // !__APPLE__

#include "kernels.hpp"
#include "check.hpp"
#include "prefix_sum.hpp"
#include "equihash.h"
#include "blake.h"

#include <map>

#define PARAM_N	200
#define PARAM_K 9
#define ZCASH_HASH_LEN 50

void GenerateHash_1(const eh_HashState& base_state, eh_index g,
                  unsigned char* hash, size_t hLen)
{
    eh_HashState state;
    state = base_state;
    eh_index lei = htole32(g);
    crypto_generichash_blake2b_update(&state, (const unsigned char*) &lei,
                                      sizeof(eh_index));
    crypto_generichash_blake2b_final(&state, hash, hLen);
}

class OpenCLSolver
{
private:
    const std::function<bool(const std::vector<unsigned char>&, const uint256& nonce)> validBlock_function;
    const std::function<bool(EhSolverCancelCheck)> cancelled_function;
    
    cl_uint max_workgroup_size;
    cl_uint init_size;
    cl_uint KeysCount;
    
    cl::Context context;
    cl::CommandQueue queue;
    
    cl::Buffer cl_blake;
    cl::Buffer cl_hashes;
    cl::Buffer cl_hash_chunks;
    cl::Buffer cl_vchunks;
    cl::Buffer cl_sieve;
    cl::Buffer cl_pair_counts;
    cl::Buffer cl_pairs;
    cl::Buffer cl_hash_chunks_1;
    cl::Buffer cl_zero_counts;
    cl::Buffer cl_partzero_one;
    cl::Buffer cl_partzero_two;
    cl::Buffer cl_indices;
    cl::Buffer cl_group_counts;
    cl::Buffer cl_schedule;
    cl::Buffer cl_group_offsets;
    cl::Buffer cl_group_partial_one;
    cl::Buffer cl_group_partial_two;
    
    std::unique_ptr<uint32_t> group_offsets = std::unique_ptr<uint32_t>(new uint32_t[15]);
    
    cl::Kernel blake_kernel;
    cl::Kernel prepare_kernel;
    cl::Kernel clear_counts_kernel;
    cl::Kernel fill_sieve_atomic;
    cl::Kernel correct_sieve_kernel;
    cl::Kernel scan_subarrays_kernel;
    cl::Kernel scan_inc_subarrays_kernel;
    cl::Kernel find_zeros_kernel;
    cl::Kernel get_indices_kernel;
    cl::Kernel group_count_kernel;
    cl::Kernel project;
    cl::Kernel group_offsets_kernel;
    cl::Kernel group2_kernel;
    cl::Kernel group3_kernel;
    cl::Kernel group4_kernel;
    cl::Kernel group5_kernel;
    cl::Kernel groupc_kernel;
    cl::Kernel final2_kernel;
    cl::Kernel final3_kernel;
    cl::Kernel final4_kernel;
    cl::Kernel final5_kernel;
    cl::Kernel finalc_kernel;
    cl::Kernel project_zeros;
    
    cl::Event blake_event;
    cl::Event prepare_event;
    
    std::vector<cl::Event> clear_counts_events;
    std::vector<cl::Event> fill_sieve_events;
    std::vector<cl::Event> subarray_events;
    std::vector<cl::Event> subarray_inc_events;
    std::vector<cl::Event> clear_schedule_events;
    std::vector<cl::Event> project_events;
    std::vector<cl::Event> group_count_events;
    std::vector<cl::Event> group_offsets_events;
    std::vector<cl::Event> group2_events;
    std::vector<cl::Event> group3_events;
    std::vector<cl::Event> group4_events;
    std::vector<cl::Event> group5_events;
    std::vector<cl::Event> groupc_events;
    cl::Event find_zeros_event;
    cl::Event get_indices_event;
    cl::Event project_zeros_event;
    
    crypto_generichash_blake2b_state init_state;
    blake2b_state_t blake_init;
    
public:
    OpenCLSolver(const std::function<bool(const std::vector<unsigned char>&, const uint256&)> validBlock_,
                 const std::function<bool(EhSolverCancelCheck)> cancelled_)
    :validBlock_function{validBlock_}, cancelled_function{cancelled_}
    {
        std::vector<cl::Platform> platforms;
        check(__LINE__,cl::Platform::get(&platforms));
        std::cout << "Number of platforms: " << platforms.size() << "\n";
        for(const auto& platform: platforms) {
            std::string name, vendor, extensions, profile, version;
            platform.getInfo(CL_PLATFORM_NAME, &name);
            platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
            platform.getInfo(CL_PLATFORM_EXTENSIONS, &extensions);
            platform.getInfo(CL_PLATFORM_PROFILE, &profile);
            platform.getInfo(CL_PLATFORM_VERSION, &version);
            std::cout << "Platform name: " << name <<
            ", vendor: " << vendor <<
            ", extensions: " << extensions <<
            ", profile: " << profile <<
            ", version: " << version << "\n";
        }
        std::cout << "\n";
        std::vector<cl::Device> devices;
        check(__LINE__,platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices));
        std::cout << "Number of GPU devices: " << devices.size() << "\n";
        bool async_supported = false;
        for(const auto& device: devices) {
            std::string name;
            cl_ulong global_mem_size;
            cl_ulong local_mem_size;
            std::string version;
            std::string profile;
            cl_ulong max_wg_size;
            cl_command_queue_properties queue_properties;
            device.getInfo(CL_DEVICE_NAME, &name);
            device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
            device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size);
            device.getInfo(CL_DEVICE_VERSION, &version);
            device.getInfo(CL_DEVICE_PROFILE, &profile);
            device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_wg_size);
            device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &queue_properties);
            if (max_workgroup_size == 0) {
                max_workgroup_size = (cl_uint)max_wg_size;
            }
            if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
            {
                async_supported = true;
            }
            std::cout << "Device name: " << name <<
            ", version: " << version <<
            ", profile: " << profile <<
            ", global mem size (MB): " << global_mem_size/1024/1024 <<
            ", local mem size (KB): " << local_mem_size/1024 <<
            ", max workgroup size: " << max_wg_size <<
            ", out-of-order: " << (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) << "\n";
        }
        if (max_workgroup_size > 256)
        {
            max_workgroup_size = 256;
        }
        context = cl::Context(devices[0]);
        queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
        
        std::cout << "Building kernels...\n";
        cl::Program::Sources sources;
        sources.push_back({kernel_source.c_str(), kernel_source.size()});
        cl::Program program(context,sources);
        if(program.build({devices[0]})!=CL_SUCCESS){
            std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])<<"\n";
            exit(1);
        }
        std::vector<cl::Kernel> kernels;
        check(__LINE__,program.createKernels(&kernels));
        for(std::size_t i = 0; i < kernels.size(); i++) {
            std::string name;
            kernels[i].getInfo(CL_KERNEL_FUNCTION_NAME, &name);
        }
        const cl_uint CollisionBitLength = 20;
        init_size = 1 << (CollisionBitLength + 1);
        KeysCount = 1 << (CollisionBitLength);
        const cl_uint scan_batch = max_workgroup_size * 2;
        const cl_uint group_segments_one = (KeysCount*15+1+scan_batch-1)/scan_batch;
        const cl_uint group_segments_two = (group_segments_one + scan_batch - 1) / scan_batch;
        
        cl_blake = cl::Buffer(context, CL_MEM_READ_WRITE, 64);
        cl_hashes = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * init_size * 32); // Initial 20
        cl_hash_chunks = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * init_size * 15); // Initial 10
        cl_vchunks = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * init_size * 2); // Initial 1
        cl_sieve = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * KeysCount * 16);
        cl_pair_counts = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * KeysCount);
        cl_pairs = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*30*init_size); // Initial 18
        cl_hash_chunks_1 = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * init_size * 15); // Initial 10
        cl_zero_counts = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*init_size * 2);
        cl_partzero_one = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*4096*2);
        cl_partzero_two = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*8*2);
        cl_indices = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*256*512);
        cl_group_counts = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*(KeysCount*15+1));
        cl_schedule = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*KeysCount);
        cl_group_offsets = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*15);
        cl_group_partial_one =cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*group_segments_one);
        cl_group_partial_two = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*group_segments_two);
        
        blake_kernel = kernelByName(kernels, "compute_blake");
        prepare_kernel = kernelByName(kernels, "prepare");
        clear_counts_kernel = kernelByName(kernels, "clear_counts");
        fill_sieve_atomic = kernelByName(kernels, "fill_sieve_atomic");
        scan_subarrays_kernel = kernelByName(kernels, "scan_subarrays");
        scan_inc_subarrays_kernel = kernelByName(kernels, "scan_inc_subarrays");
        find_zeros_kernel = kernelByName(kernels, "find_zeros");
        get_indices_kernel = kernelByName(kernels, "get_indices");
        group_count_kernel = kernelByName(kernels, "group_count");
        project = kernelByName(kernels, "project");
        group_offsets_kernel = kernelByName(kernels, "group_offsets");
        group2_kernel = kernelByName(kernels, "group2");
        group3_kernel = kernelByName(kernels, "group3");
        group4_kernel = kernelByName(kernels, "group4");
        group5_kernel = kernelByName(kernels, "group5");
        groupc_kernel = kernelByName(kernels, "group_c");
        final2_kernel = kernelByName(kernels, "final2");
        final3_kernel = kernelByName(kernels, "final3");
        final4_kernel = kernelByName(kernels, "final4");
        final5_kernel = kernelByName(kernels, "final5");
        finalc_kernel = kernelByName(kernels, "final_c");
        project_zeros = kernelByName(kernels, "project_zeros");
        
        Eh200_9.InitialiseState(init_state);
        zcash_blake2b_init(&blake_init, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
    }

private:
    cl_uint solution_cycle(
                           cl::Buffer& cl_hash_chunks_src,
                           cl::Buffer& cl_hash_chunks_dst,
                           const cl_uint chunks,
                           const cl_uint problem_size,
                           const cl_uint pairs_prev_offset,
                           const cl_uint pairs_offset
                           )
    {

        if (chunks < 9) {
            check(__LINE__,clear_counts_kernel.setArg(0, cl_pair_counts));
            clear_counts_events.emplace_back();
            check(__LINE__,queue.enqueueNDRangeKernel(clear_counts_kernel, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &clear_counts_events.back()));

            check(__LINE__,fill_sieve_atomic.setArg(0, cl_vchunks));
            check(__LINE__,fill_sieve_atomic.setArg(1, cl_sieve));
            check(__LINE__,fill_sieve_atomic.setArg(2, cl_pair_counts));
            fill_sieve_events.emplace_back();
            check(__LINE__,queue.enqueueNDRangeKernel(fill_sieve_atomic, cl::NullRange,cl::NDRange{problem_size},cl::NullRange, nullptr, &fill_sieve_events.back()));
        }

        check(__LINE__,group_count_kernel.setArg(0, cl_pair_counts));
        check(__LINE__,group_count_kernel.setArg(1, cl_group_counts));
        group_count_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group_count_kernel, cl::NullRange, cl::NDRange{KeysCount,15}, cl::NDRange{16,15}, nullptr, &group_count_events.back()));
        prefix_sum(queue, cl_group_counts, KeysCount*15+1, max_workgroup_size, cl_group_partial_one, cl_group_partial_two, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events);
        check(__LINE__,group_offsets_kernel.setArg(0, cl_group_counts));
        check(__LINE__,group_offsets_kernel.setArg(1, cl_group_offsets));
        check(__LINE__,group_offsets_kernel.setArg(2, KeysCount));
        group_offsets_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group_offsets_kernel, cl::NullRange, cl::NDRange{15}, cl::NullRange, nullptr, &group_offsets_events.back()));
        check(__LINE__,queue.finish());
        check(__LINE__,queue.enqueueReadBuffer(cl_group_offsets, CL_TRUE, 0, sizeof(uint32_t)*15, group_offsets.get()));
        check(__LINE__,project.setArg(0, cl_pair_counts));
        check(__LINE__,project.setArg(1, cl_group_counts));
        check(__LINE__,project.setArg(2, cl_schedule));
        project_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(project, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &project_events.back()));
        cl_uint dest_offset = 0;
        cl_uint size = group_offsets.get()[0];
        cl_uint schedule_offset = 0;
        cl_uint num_groups = max_workgroup_size/chunks;
        check(__LINE__,group2_kernel.setArg(0, cl_schedule));
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
        check(__LINE__,group2_kernel.setArg(11, cl_vchunks));
        group2_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group2_kernel, cl::NDRange{0}, cl::NDRange{chunks*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{num_groups*chunks}, nullptr, &group2_events.back()));
        dest_offset += 1*size;
        schedule_offset += size;
        size = group_offsets.get()[1];
        num_groups = max_workgroup_size/chunks/3;
        check(__LINE__,group3_kernel.setArg(0, cl_schedule));
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
        check(__LINE__,group3_kernel.setArg(14, cl_vchunks));
        group3_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group3_kernel, cl::NullRange, cl::NDRange{(3*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{3*num_groups*chunks}, nullptr, &group3_events.back()));
        dest_offset += 3*size;
        schedule_offset += size;
        size = group_offsets.get()[2];
        num_groups = max_workgroup_size/chunks/4;
        check(__LINE__,group4_kernel.setArg(0, cl_schedule));
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
        check(__LINE__,group4_kernel.setArg(14, cl_vchunks));
        group4_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group4_kernel, cl::NullRange, cl::NDRange{(4*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{4*num_groups*chunks}, nullptr, &group4_events.back()));
        dest_offset += 6*size;
        schedule_offset += size;
        size = group_offsets.get()[3];
        num_groups = max_workgroup_size/chunks/5;
        check(__LINE__,group5_kernel.setArg(0, cl_schedule));
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
        check(__LINE__,group5_kernel.setArg(14, cl_vchunks));
        group5_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group5_kernel, cl::NullRange, cl::NDRange{(5*chunks)*num_groups*((size+num_groups-1)/num_groups)}, cl::NDRange{5*num_groups*chunks}, nullptr, &group5_events.back()));
        dest_offset += 10*size;
        schedule_offset += size;
        check(__LINE__,groupc_kernel.setArg(0, cl_schedule));
        check(__LINE__,groupc_kernel.setArg(3, cl_sieve));
        check(__LINE__,groupc_kernel.setArg(4, cl_hash_chunks_src));
        check(__LINE__,groupc_kernel.setArg(5, cl_hash_chunks_dst));
        check(__LINE__,groupc_kernel.setArg(6, cl_pairs));
        check(__LINE__,groupc_kernel.setArg(7, pairs_prev_offset));
        check(__LINE__,groupc_kernel.setArg(8, pairs_offset));
        check(__LINE__,groupc_kernel.setArg(10, (cl_uchar)chunks));
        check(__LINE__,groupc_kernel.setArg(15, cl_vchunks));
        
        for(uint count = 6;count<=16;++count) {
            size = group_offsets.get()[count-2];
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
        return dest_offset;
    }

    cl_uint final_cycle(
        cl::Buffer& cl_hash_chunks_src,
        const cl_uint problem_size,
        const cl_uint pairs_prev_offset,
        const cl_uint pairs_offset)
    {
        check(__LINE__,clear_counts_kernel.setArg(0, cl_pair_counts));
        clear_counts_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(clear_counts_kernel, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &clear_counts_events.back()));
        
        check(__LINE__,fill_sieve_atomic.setArg(0, cl_vchunks));
        check(__LINE__,fill_sieve_atomic.setArg(1, cl_sieve));
        check(__LINE__,fill_sieve_atomic.setArg(2, cl_pair_counts));
        fill_sieve_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(fill_sieve_atomic, cl::NullRange,cl::NDRange{problem_size},cl::NullRange, nullptr, &fill_sieve_events.back()));
        check(__LINE__,group_count_kernel.setArg(0, cl_pair_counts));
        check(__LINE__,group_count_kernel.setArg(1, cl_group_counts));
        group_count_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group_count_kernel, cl::NullRange, cl::NDRange{KeysCount,15}, cl::NDRange{16,15}, nullptr, &group_count_events.back()));
        prefix_sum(queue, cl_group_counts, KeysCount*15+1, max_workgroup_size, cl_group_partial_one, cl_group_partial_two, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events);
        check(__LINE__,group_offsets_kernel.setArg(0, cl_group_counts));
        check(__LINE__,group_offsets_kernel.setArg(1, cl_group_offsets));
        check(__LINE__,group_offsets_kernel.setArg(2, KeysCount));
        group_offsets_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(group_offsets_kernel, cl::NullRange, cl::NDRange{15}, cl::NullRange, nullptr, &group_offsets_events.back()));
        check(__LINE__,queue.enqueueReadBuffer(cl_group_offsets, CL_TRUE, 0, sizeof(uint32_t)*15, group_offsets.get()));
        check(__LINE__,project.setArg(0, cl_pair_counts));
        check(__LINE__,project.setArg(1, cl_group_counts));
        check(__LINE__,project.setArg(2, cl_schedule));
        project_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(project, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &project_events.back()));
        cl_uint dest_offset = 0;
        cl_uint size = group_offsets.get()[0];
        cl_uint schedule_offset = 0;
        cl_uint num_groups = max_workgroup_size;
        check(__LINE__,final2_kernel.setArg(0, cl_schedule));
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
        size = group_offsets.get()[1];
        num_groups = max_workgroup_size/3;
        check(__LINE__,final3_kernel.setArg(0, cl_schedule));
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
        size = group_offsets.get()[2];
        num_groups = max_workgroup_size/4;
        check(__LINE__,final4_kernel.setArg(0, cl_schedule));
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
        size = group_offsets.get()[3];
        num_groups = max_workgroup_size/5;
        check(__LINE__,final5_kernel.setArg(0, cl_schedule));
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
        check(__LINE__,finalc_kernel.setArg(0, cl_schedule));
        check(__LINE__,finalc_kernel.setArg(3, cl_sieve));
        check(__LINE__,finalc_kernel.setArg(4, cl_hash_chunks_src));
        check(__LINE__,finalc_kernel.setArg(5, cl_pairs));
        check(__LINE__,finalc_kernel.setArg(6, pairs_prev_offset));
        check(__LINE__,finalc_kernel.setArg(7, pairs_offset));
        
        for(uint count = 6;count<=16;++count) {
            size = group_offsets.get()[count-2];
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

public:
        
    bool run(const uint8_t* header_) // header must be 108 bytes long
    {
    
        uint8_t header[140];
        std::fill(header, header + 140, 0);
        std::copy(header_, header_ + 108, header);
        for(uint32_t v = 0; v < 10000; ++v)
        {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            fill_sieve_events.clear();
            subarray_events.clear();
            subarray_inc_events.clear();
            clear_schedule_events.clear();
            project_events.clear();
            group_count_events.clear();
            group_offsets_events.clear();
            group2_events.clear();
            group3_events.clear();
            group4_events.clear();
            group5_events.clear();
            groupc_events.clear();
            clear_counts_events.clear();
            
            char s[16];
            sprintf(s, "%x", v);
            uint256 V = uint256S(s);
            std::copy(V.begin(), V.begin()+20, header + 108);
            blake2b_state_t blake{blake_init};
            zcash_blake2b_update(&blake, header, 128, 0);
            crypto_generichash_blake2b_state state{init_state}; // CPU blake
            crypto_generichash_blake2b_update(&state, header, 140);
            check(__LINE__,queue.enqueueWriteBuffer(cl_blake, CL_TRUE, 0, sizeof(blake.h), blake.h));
            check(__LINE__,blake_kernel.setArg(0, cl_blake));
            check(__LINE__,blake_kernel.setArg(1, cl_hashes));
            check(__LINE__,queue.enqueueNDRangeKernel(blake_kernel, cl::NullRange,cl::NDRange{init_size/2},cl::NDRange{64}, nullptr, &blake_event));
            check(__LINE__,clear_counts_kernel.setArg(0, cl_pair_counts));
            clear_counts_events.emplace_back();
            check(__LINE__,queue.enqueueNDRangeKernel(clear_counts_kernel, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange, nullptr, &clear_counts_events.back()));
            check(__LINE__,prepare_kernel.setArg(0, cl_hashes));
            check(__LINE__,prepare_kernel.setArg(1, cl_hash_chunks));
            check(__LINE__,prepare_kernel.setArg(2, cl_sieve));
            check(__LINE__,prepare_kernel.setArg(3, cl_pair_counts));
            check(__LINE__,queue.enqueueNDRangeKernel(prepare_kernel, cl::NullRange,cl::NDRange{init_size/2, 20},cl::NDRange{1, 20}, nullptr, &prepare_event));
            cl_uint problem_size = init_size;
            cl_uint pair_offset = 0;
            cl_uint pair_prev_offset = 0;
            for (cl_uint i=9; i>=2; --i) {
                //std::cout << std::dec << "Problem size =" << problem_size << "\n";
                problem_size = solution_cycle((i%2==0)?cl_hash_chunks_1:cl_hash_chunks, (i%2==0)?cl_hash_chunks:cl_hash_chunks_1, i, problem_size, pair_prev_offset, pair_offset);
                if (problem_size > 2500000)
                {
                    problem_size = 2500000;
                }
                pair_prev_offset = pair_offset;
                pair_offset += problem_size;
            }
            problem_size = final_cycle(cl_hash_chunks, problem_size, pair_prev_offset, pair_offset);
            
            check(__LINE__,find_zeros_kernel.setArg(0, cl_pairs));
            check(__LINE__,find_zeros_kernel.setArg(1, pair_offset));
            check(__LINE__,find_zeros_kernel.setArg(2, cl_zero_counts));
            check(__LINE__,queue.enqueueNDRangeKernel(find_zeros_kernel, cl::NullRange, cl::NDRange{problem_size}, cl::NullRange, nullptr, &find_zeros_event));
            
            prefix_sum(queue, cl_zero_counts, problem_size + 1, max_workgroup_size, cl_partzero_one, cl_partzero_two, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events);
            
            check(__LINE__,project_zeros.setArg(0, cl_zero_counts));
            check(__LINE__,project_zeros.setArg(1, cl_schedule));
            
            check(__LINE__,queue.enqueueNDRangeKernel(project_zeros, cl::NullRange, cl::NDRange{problem_size}, cl::NullRange, nullptr, &project_zeros_event));
            check(__LINE__,queue.finish());

            std::unique_ptr<uint32_t> zero_counts = std::unique_ptr<uint32_t>(new uint32_t[problem_size+1]);
            check(__LINE__,queue.enqueueReadBuffer(cl_zero_counts, CL_TRUE, 0, sizeof(cl_uint)*(problem_size+1), zero_counts.get()));
            size_t zeros = zero_counts.get()[problem_size];
            if (zeros > 256) {
                zeros = 256;
            }
            
            check(__LINE__,get_indices_kernel.setArg(0, cl_schedule));
            check(__LINE__,get_indices_kernel.setArg(1, cl_pairs));
            check(__LINE__,get_indices_kernel.setArg(2, pair_offset));
            check(__LINE__,get_indices_kernel.setArg(3, cl_indices));
            check(__LINE__,queue.enqueueNDRangeKernel(get_indices_kernel, cl::NullRange, cl::NDRange{zeros*256}, cl::NDRange{256}, nullptr, &get_indices_event));
            check(__LINE__,queue.finish());
            
            //std::cout << "Zeros after scan: " << zero_counts.get()[problem_size] << "\n";
            std::unique_ptr<uint32_t> indices = std::unique_ptr<uint32_t>(new uint32_t[256*512]);
            check(__LINE__,queue.enqueueReadBuffer(cl_indices, CL_TRUE, 0, sizeof(cl_uint)*256*512, indices.get()));

            size_t distinct_sets = 0;
            unsigned char tmpHash[50];
            const size_t FullWidth=2*3+sizeof(eh_index)*(1 << (PARAM_K-1));
            const size_t FinalFullWidth=2*3+sizeof(eh_index)*(1 << (PARAM_K));
            const size_t CollisionByteLength=3;
            for(size_t i=0;i<zeros;i++) {
                std::set<uint32_t> index_set;
                for(size_t j=0;j<512;j++)
                {
                    index_set.insert(indices.get()[512*i+j]);
                }
                if (index_set.size() != 512) {
                    continue;
                }
                distinct_sets++;
                std::vector<FullStepRow<FullWidth>> Y1;
                for(size_t j=0;j<256;j++) {
                    uint32_t i1 = indices.get()[512*i+j*2];
                    uint32_t i2 = indices.get()[512*i+j*2+1];
                    GenerateHash_1(state, (i1/2), tmpHash, 50);
                    FullStepRow<FullWidth> X1{tmpHash+((i1%2)*200/8), (size_t)200/8, (size_t)30, (size_t)20,  (eh_index)i1};
                    GenerateHash_1(state, (i2/2), tmpHash, 50);
                    FullStepRow<FullWidth> X2{tmpHash+((i2%2)*200/8), (size_t)200/8, (size_t)30, (size_t)20,  (eh_index)i2};
                    //assert(HasCollision(X1, X2, CollisionByteLength));
                    Y1.emplace_back(X1, X2, 30, 4, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y2;
                for(size_t j=0;j<128;j++) {
                    //assert(HasCollision(Y1[j*2], Y1[j*2+1], CollisionByteLength));
                    Y2.emplace_back(Y1[j*2], Y1[j*2+1], 27, 8, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y3;
                for(size_t j=0;j<64;j++) {
                    //assert(HasCollision(Y2[j*2], Y2[j*2+1], CollisionByteLength));
                    Y3.emplace_back(Y2[j*2], Y2[j*2+1], 24, 16, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y4;
                for(size_t j=0;j<32;j++) {
                    //assert(HasCollision(Y3[j*2], Y3[j*2+1], CollisionByteLength));
                    Y4.emplace_back(Y3[j*2], Y3[j*2+1], 21, 32, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y5;
                for(size_t j=0;j<16;j++) {
                    //assert(HasCollision(Y4[j*2], Y4[j*2+1], CollisionByteLength));
                    Y5.emplace_back(Y4[j*2], Y4[j*2+1], 18, 64, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y6;
                for(size_t j=0;j<8;j++) {
                    //assert(HasCollision(Y5[j*2], Y5[j*2+1], CollisionByteLength));
                    Y6.emplace_back(Y5[j*2], Y5[j*2+1], 15, 128, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y7;
                for(size_t j=0;j<4;j++) {
                    //assert(HasCollision(Y6[j*2], Y6[j*2+1], CollisionByteLength));
                    Y7.emplace_back(Y6[j*2], Y6[j*2+1], 12, 256, 3);
                }
                std::vector<FullStepRow<FullWidth>> Y8;
                for(size_t j=0;j<2;j++) {
                    //assert(HasCollision(Y7[j*2], Y7[j*2+1], CollisionByteLength));
                    Y8.emplace_back(Y7[j*2], Y7[j*2+1], 9, 512, 3);
                }
                assert(HasCollision(Y8[0], Y8[1], CollisionByteLength*2));
                if (DistinctIndices(Y8[0], Y8[1], 6, 1024)) {
                    FullStepRow<FinalFullWidth> res(Y8[0], Y8[1], 6, 1024, 0);
                    auto soln = res.GetIndices(6, 2*1024, 20);
                    //assert(soln.size() == equihash_solution_size(200, 9));
                    if (validBlock_function(soln, V)) {
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                        std::cout << "ROUND in us = " << std::dec << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() <<std::endl;
                        return true;
                    }
                }
            }
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "ROUND in us = " << std::dec << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() <<std::endl;
        }
        return false;
    }
     
    void print_timings()
    {
        cl_ulong blake_time = blake_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - blake_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "Blake took " << blake_time << " ns\n";
        cl_ulong prepare_time = prepare_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prepare_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "Prepare took " << prepare_time << " ns\n";
        std::cout << " ns\n";
        std::cout << "Clear counts took ";
        for(const auto& event: clear_counts_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Fill sieve took ";
        for(const auto& event: fill_sieve_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Subarray scan took ";
        for(const auto& event: subarray_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Subarray inc scan took ";
        for(const auto& event: subarray_inc_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Project took ";
        for(const auto& event: project_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group count took ";
        for(const auto& event: group_count_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group offsets took ";
        for(const auto& event: group_offsets_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group 2 took ";
        for(const auto& event: group2_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group 3 took ";
        for(const auto& event: group3_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group 4 took ";
        for(const auto& event: group4_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group 5 took ";
        for(const auto& event: group5_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        std::cout << "Group C took ";
        for(const auto& event: groupc_events) {
            cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            std::cout << time << " ";
        }
        std::cout << " ns\n";
        cl_ulong find_zeros_time = find_zeros_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - find_zeros_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "Find zeros took " << find_zeros_time << " ns\n";
        cl_ulong project_zeros_time = project_zeros_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - find_zeros_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "Project zeros took " << project_zeros_time << " ns\n";
        cl_ulong get_indices_time = get_indices_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - get_indices_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << "Get indices took " << get_indices_time << " ns\n";
    }
};

#endif /* opencl_solve_h */
