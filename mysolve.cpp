//
//  mysolve.cpp
//  FXTrue
//
//  Created by Alexey Akhunov on 31/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#include "mysolve.hpp"

#include "blake.h"
#include "uint256.h"
#include "equihash.h"
#include <map>

#if defined(__APPLE__) || defined(__MACOSX)
#include "cl.hpp"
#else
#include <CL/cl.hpp>
#endif // !__APPLE__

#include "kernels.hpp"
#include "check.hpp"
#include "prefix_sum.hpp"
#include "solution_cycle.hpp"

#define PARAM_N	200
#define PARAM_K 9
#define ZCASH_HASH_LEN 50

bool opencl_Solve(const uint256& V,
                  const eh_HashState& base_state,
                  const std::function<bool(std::vector<unsigned char>)> validBlock,
                  const std::function<bool(EhSolverCancelCheck)> cancelled)
{
    blake2b_state_t blake_init;
    zcash_blake2b_init(&blake_init, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
    uint8_t header[140];
    std::fill(header, header + 140, 0);
    std::copy(V.begin(), V.begin()+20, header + 108);
    std::cout << "Header for GPU: ";
    for (int i=0;i<140;i++)
    {
        unsigned int c1 = header[i];
        std::cout << std::hex << c1 << " ";
    }
    blake2b_state_t blake{blake_init};
    zcash_blake2b_update(&blake, header, 128, 0);
    std::cout << "Blake state GPU ";
    for(int i=0;i<8;i++) {
        std::cout << i << ": " << std::hex << blake.h[i] << " ";
    }
    std::cout << "\n" << std::dec;
    std::cout << "Blake state CPU ";
    auto blake_cpu = base_state;
    for(int i=0;i<8;i++) {
        std::cout << i << ": " << std::hex << blake_cpu.h[i] << " ";
    }
    std::cout << "\n" << std::dec;
    // Querying platforms
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
    cl_uint max_workgroup_size = 0;
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
    std::cout << "\n";
    cl::Context context{devices[0]};
    //cl::CommandQueue async(context, devices[0], CL_QUEUE_PROFILING_ENABLE | (async_supported ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
    
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
        //std::cout << "Kernel "<< name << "\n";
    }
    cl::Buffer cl_blake(context, CL_MEM_READ_WRITE, sizeof(blake.h));
    cl::Kernel& blake_kernel = kernelByName(kernels, "compute_blake");
    
    std::cout << "Generating first list\n";
    const cl_uint CollisionBitLength = 20;
    cl_uint init_size { 1 << (CollisionBitLength + 1) };
    std::unique_ptr<unsigned char> hashes = std::unique_ptr<unsigned char>(new unsigned char[init_size*32]); // 64 bytes per hash but 1 hash per 2 indices
    for (eh_index g = 0, hi = 0; g < init_size/2; g++, hi+=64) {
        GenerateHash(base_state, g, hashes.get()+hi, 50);
        if (cancelled(ListGeneration)) throw solver_cancelled;
    }
    //cl::Buffer cl_hashes(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * init_size * 32, hashes.get());
    cl::Buffer cl_hashes(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * init_size * 32);
    cl::Buffer cl_hash_chunks(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * init_size * 15);
    cl::Kernel& prepare_kernel = kernelByName(kernels, "prepare");
    const cl_uint KeysCount = 1 << (CollisionBitLength);
    cl::Buffer cl_sieve(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * KeysCount * 16);
    cl::Kernel& clear_sieve_kernel = kernelByName(kernels, "clear_sieve");
    cl::Kernel& fill_sieve_kernel = kernelByName(kernels, "fill_sieve");
    cl::Kernel& correct_sieve_kernel = kernelByName(kernels, "correct_sieve");
    cl::Buffer cl_pair_counts(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * KeysCount);
    cl::Kernel& align_collisions_kernel = kernelByName(kernels, "align_collisions");
    const cl_uint scan_batch = max_workgroup_size * 2;
    const cl_uint segments_one = (KeysCount+scan_batch-1)/scan_batch;
    cl::Buffer cl_partial_one(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * segments_one);
    cl::Kernel& scan_subarrays_kernel = kernelByName(kernels, "scan_subarrays");
    const cl_uint segments_two = (segments_one + scan_batch - 1) / scan_batch;
    cl::Buffer cl_partial_two(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * segments_two);
    cl::Kernel& scan_inc_subarrays_kernel = kernelByName(kernels, "scan_inc_subarrays");
    cl::Buffer cl_schedule(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * init_size);
    cl::Buffer cl_pairs(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*30*init_size);
    cl::Buffer cl_hash_chunks_1(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * init_size * 15);
    cl::Buffer cl_zero_counts(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*init_size * 2);
    cl::Buffer cl_partzero_one(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*4096*2);
    cl::Buffer cl_partzero_two(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*8*2);
    cl::Kernel& find_zeros_kernel = kernelByName(kernels, "find_zeros");
    cl::Buffer cl_indices(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*256*512);
    cl::Kernel& get_indices_kernel = kernelByName(kernels, "get_indices");
    // Counting how many groups of 2, 3, 4, 5, ..., 16 collisions there are
    cl::Buffer cl_group_counts(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*(KeysCount*15+1));
    cl::Kernel& group_count_kernel = kernelByName(kernels, "group_count");
    cl::Buffer cl_schedule_v2(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*KeysCount);
    cl::Kernel& project_v2 = kernelByName(kernels, "project_v2");
    cl::Kernel& clear_schedule_v2 = kernelByName(kernels, "clear_schedule_v2");
    cl::Buffer cl_group_offsets(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*15);
    cl::Kernel& group_offsets_kernel = kernelByName(kernels, "group_offsets");
    cl::Kernel& group2_kernel = kernelByName(kernels, "group2");
    cl::Kernel& group3_kernel = kernelByName(kernels, "group3");
    cl::Kernel& group4_kernel = kernelByName(kernels, "group4");
    cl::Kernel& group5_kernel = kernelByName(kernels, "group5");
    cl::Kernel& groupc_kernel = kernelByName(kernels, "group_c");
    cl::Kernel& final2_kernel = kernelByName(kernels, "final2");
    cl::Kernel& final3_kernel = kernelByName(kernels, "final3");
    cl::Kernel& final4_kernel = kernelByName(kernels, "final4");
    cl::Kernel& final5_kernel = kernelByName(kernels, "final5");
    cl::Kernel& finalc_kernel = kernelByName(kernels, "final_c");
    
    std::unique_ptr<uint32_t> group_counts = std::unique_ptr<uint32_t>(new uint32_t[KeysCount*15+1]);
    std::unique_ptr<uint32_t> group_offsets = std::unique_ptr<uint32_t>(new uint32_t[15]);
    std::unique_ptr<uint32_t> schedule_v2 = std::unique_ptr<uint32_t>(new uint32_t[KeysCount]);
    
    check(__LINE__,queue.enqueueWriteBuffer(cl_blake, CL_TRUE, 0, sizeof(blake.h), blake.h));
    check(__LINE__,blake_kernel.setArg(0, cl_blake));
    check(__LINE__,blake_kernel.setArg(1, cl_hashes));
    check(__LINE__,queue.enqueueNDRangeKernel(blake_kernel, cl::NullRange,cl::NDRange{init_size/2},cl::NDRange{64}));
    check(__LINE__,queue.finish());
    std::unique_ptr<unsigned char> hashes_1 = std::unique_ptr<unsigned char>(new unsigned char[init_size*32]);
    check(__LINE__,queue.enqueueReadBuffer(cl_hashes, CL_TRUE, 0, init_size*32, hashes_1.get()));
    for (int i=0;i<64;i++)
    {
        unsigned int c1 = hashes.get()[i];
        std::cout << std::hex << c1 << " ";
    }
    std::cout << "\n";
    for (int i=0;i<64;i++)
    {
        unsigned int c2 = hashes_1.get()[i];
        std::cout << std::hex << c2 << " ";
    }
    std::cout << "\n";
    std::cout << std::dec;
    
    check(__LINE__,prepare_kernel.setArg(0, cl_hashes));
    check(__LINE__,prepare_kernel.setArg(1, cl_hash_chunks));
    check(__LINE__,queue.enqueueNDRangeKernel(prepare_kernel, cl::NullRange,cl::NDRange{init_size/2, 20},cl::NDRange{1, 20}));
    check(__LINE__,queue.finish());
    std::unique_ptr<uint32_t> hash_chunks = std::unique_ptr<uint32_t>(new uint32_t[init_size*10]);
    check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks, CL_TRUE, 0, sizeof(uint32_t)*init_size*10, hash_chunks.get()));
    
    // Compare with the basic solve
    const size_t CollisionByteLength=(CollisionBitLength+7)/8;
    const size_t K = 9;
    const size_t N = 200;
    const size_t HashLength=(K+1)*CollisionByteLength;
    const size_t FullWidth=2*CollisionByteLength+sizeof(eh_index)*(1 << (K-1));
    std::vector<FullStepRow<FullWidth>> X;
    X.reserve(init_size);
    const size_t IndicesPerHashOutput=512/N;
    const size_t HashOutput=IndicesPerHashOutput*N/8;
    unsigned char tmpHash[HashOutput];
    for (eh_index g = 0; X.size() < init_size; g++) {
        GenerateHash(base_state, g, tmpHash, HashOutput);
        for (eh_index i = 0; i < IndicesPerHashOutput && X.size() < init_size; i++) {
            X.emplace_back(tmpHash+(i*N/8), N/8, HashLength,
                           CollisionBitLength, (g*IndicesPerHashOutput)+i);
        }
        if (cancelled(ListGeneration)) throw solver_cancelled;
    }
    
    // Populate the sieve
    
    check(__LINE__,clear_sieve_kernel.setArg(0, cl_sieve));
    check(__LINE__,queue.enqueueNDRangeKernel(clear_sieve_kernel, cl::NullRange, cl::NDRange{KeysCount*16}));
    check(__LINE__,queue.finish());
    std::unique_ptr<uint32_t> sieve = std::unique_ptr<uint32_t>(new uint32_t[KeysCount * 16]);
    check(__LINE__,queue.enqueueReadBuffer(cl_sieve, CL_TRUE, 0, sizeof(cl_uint) * KeysCount * 16, sieve.get()));
    int count_non_empty = 0;
    for(size_t i=0;i<KeysCount*16;i++) {
        if (sieve.get()[i] != 0xFFFFFFFFU) {
            count_non_empty++;
        }
    }
    std::cout << "Non-empty sieve cells after cleaning: " << count_non_empty << "\n";
    
    
    check(__LINE__,fill_sieve_kernel.setArg(0, cl_hash_chunks));
    check(__LINE__,fill_sieve_kernel.setArg(1, cl_sieve));
    check(__LINE__,fill_sieve_kernel.setArg(2, (cl_uchar)10));
    check(__LINE__,queue.enqueueNDRangeKernel(fill_sieve_kernel, cl::NullRange,cl::NDRange{init_size},cl::NullRange));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_sieve, CL_TRUE, 0, sizeof(cl_uint) * KeysCount * 16, sieve.get()));
    count_non_empty = 0;
    for(size_t i=0;i<KeysCount*16;i++) {
        if (sieve.get()[i] != 0xFFFFFFFFU) {
            count_non_empty++;
        }
    }
    std::cout << "Non-empty sieve cells after filling: " << count_non_empty << "\n";
    std::cout << "Pair loss: " << (init_size - count_non_empty) << "\n";
    // Check that hashes in all rows collide
    for(size_t row=0;row<KeysCount;row++) {
        size_t first_index = 0;
        bool found_first = false;
        for(size_t col=0;col<16;col++) {
            uint32_t index = sieve.get()[row*16+col];
            if (index != 0xFFFFFFFFU) {
                if (!found_first) {
                    found_first = true;
                    first_index = index;
                } else {
                    assert(HasCollision(X[index], X[first_index], CollisionByteLength));
                }
            }
        }
    }
    
    check(__LINE__,correct_sieve_kernel.setArg(0, cl_hash_chunks));
    check(__LINE__,correct_sieve_kernel.setArg(1, cl_sieve));
    check(__LINE__,correct_sieve_kernel.setArg(2, (cl_uchar)10));
    check(__LINE__,queue.enqueueNDRangeKernel(correct_sieve_kernel, cl::NullRange,cl::NDRange{init_size},cl::NullRange));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_sieve, CL_TRUE, 0, sizeof(cl_uint) * KeysCount * 16, sieve.get()));
    count_non_empty = 0;
    for(size_t i=0;i<KeysCount*16;i++) {
        if (sieve.get()[i] != 0xFFFFFFFFU) {
            count_non_empty++;
        }
    }
    std::cout << "Non-empty sieve cells after correction: " << count_non_empty << "\n";
    std::cout << "Sieve loss after correction: " << (init_size - count_non_empty) << "\n";
    // Check that hashes in all rows collide
    for(size_t row=0;row<KeysCount;row++) {
        size_t first_index = 0;
        bool found_first = false;
        for(size_t col=0;col<16;col++) {
            uint32_t index = sieve.get()[row*16+col];
            if (index != 0xFFFFFFFFU) {
                if (!found_first) {
                    found_first = true;
                    first_index = index;
                } else {
                    assert(HasCollision(X[index], X[first_index], CollisionByteLength));
                }
            }
        }
    }
    
    // Align collisions
    
    check(__LINE__,align_collisions_kernel.setArg(0, cl_sieve));
    check(__LINE__,align_collisions_kernel.setArg(1, cl_pair_counts));
    check(__LINE__,queue.enqueueNDRangeKernel(align_collisions_kernel, cl::NullRange, cl::NDRange{16,KeysCount}, cl::NDRange{16, 16}));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_sieve, CL_TRUE, 0, sizeof(cl_uint) * KeysCount * 16, sieve.get()));
    std::unique_ptr<uint32_t> pair_counts = std::unique_ptr<uint32_t>(new uint32_t[KeysCount]);
    check(__LINE__,queue.enqueueReadBuffer(cl_pair_counts, CL_TRUE, 0, sizeof(cl_uint) * KeysCount, pair_counts.get()));
    count_non_empty = 0;
    for(size_t i=0;i<KeysCount;i++) {
        if (pair_counts.get()[i] > 0) {
            uint32_t pair_count = pair_counts.get()[i]&0xFFFFFFU;
            size_t count = 0;
            count_non_empty++;
            for(size_t j=1;count<pair_count;j++,count += j, count_non_empty++) {
                assert(HasCollision(X[sieve.get()[i*16+j]], X[sieve.get()[i*16]], CollisionByteLength));
            }
        }
    }
    std::cout << "Non-empty sieve cells after alignment: " << count_non_empty << "\n";
    
    check(__LINE__,group_count_kernel.setArg(0, cl_pair_counts));
    check(__LINE__,group_count_kernel.setArg(1, cl_group_counts));
    check(__LINE__,queue.enqueueNDRangeKernel(group_count_kernel, cl::NullRange, cl::NDRange{KeysCount,15}, cl::NDRange{16,15}));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_group_counts, CL_TRUE, 0, sizeof(uint32_t)*(KeysCount*15+1), group_counts.get()));
    std::map<int, int> group_stats;
    for(int i = 0;i<KeysCount;++i)
    {
        for(int c = 2;c<=16;++c)
        {
            group_stats[c] += group_counts.get()[(c-2)*KeysCount+i];
        }
    }
    std::cout << "Last of the group counts: " << group_counts.get()[KeysCount*15] << "\n";
    std::cout << "- Group sizes: ";
    count_non_empty = 0;
    for(const auto& i: group_stats) {
        std::cout << i.first << ": " << i.second << ", ";
        count_non_empty += i.first * i.second;
    }
    std::cout << "\n";
    std::cout << "Total: " << count_non_empty << "\n";
    
    const cl_uint group_segments_one = (KeysCount*15+1+scan_batch-1)/scan_batch;
    const cl_uint group_segments_two = (group_segments_one + scan_batch - 1) / scan_batch;
    cl::Buffer cl_group_partial_one(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*group_segments_one);
    cl::Buffer cl_group_partial_two(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*group_segments_two);
    std::vector<cl::Event> e1;
    std::vector<cl::Event> e2;
    prefix_sum(queue, cl_group_counts, KeysCount*15+1, max_workgroup_size, cl_group_partial_one, cl_group_partial_two, scan_subarrays_kernel, e1, scan_inc_subarrays_kernel, e2);
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_group_counts, CL_TRUE, 0, sizeof(uint32_t)*(KeysCount*15+1), group_counts.get()));
    std::cout << "Last of the group counts after scan: " << group_counts.get()[KeysCount*15] << "\n";
    
    check(__LINE__,group_offsets_kernel.setArg(0, cl_group_counts));
    check(__LINE__,group_offsets_kernel.setArg(1, cl_group_offsets));
    check(__LINE__,group_offsets_kernel.setArg(2, KeysCount));
    check(__LINE__,queue.enqueueNDRangeKernel(group_offsets_kernel, cl::NullRange, cl::NDRange{15}, cl::NullRange));
    
    check(__LINE__,queue.enqueueReadBuffer(cl_group_offsets, CL_TRUE, 0, sizeof(uint32_t)*15, group_offsets.get()));
    std::cout << "Group offsets: ";
    for(int i=0;i<15;i++) {
        std::cout << group_offsets.get()[i] << " ";
    }
    std::cout << "\n";
    
    check(__LINE__,clear_schedule_v2.setArg(0, cl_schedule_v2));
    check(__LINE__,queue.enqueueNDRangeKernel(clear_schedule_v2, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange));
    check(__LINE__,queue.finish());
    check(__LINE__,project_v2.setArg(0, cl_pair_counts));
    check(__LINE__,project_v2.setArg(1, cl_group_counts));
    check(__LINE__,project_v2.setArg(2, cl_schedule_v2));
    check(__LINE__,queue.enqueueNDRangeKernel(project_v2, cl::NullRange, cl::NDRange{KeysCount}, cl::NullRange));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_schedule_v2, CL_TRUE, 0, sizeof(uint32_t)*KeysCount, schedule_v2.get()));
    
    count_non_empty = 0;
    for(size_t i=0;i<KeysCount;++i)
    {
        if (schedule_v2.get()[i] != 0xFFFFFFFFU) {
            count_non_empty++;
        }
    }
    std::cout << "Schedule_v2 size: " << count_non_empty << "\n";
    
    cl_uint dest_offset = 0;
    cl_uint size = group_offsets.get()[0];
    cl_uint schedule_offset = 0;
    check(__LINE__,group2_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group2_kernel.setArg(1, size));
    check(__LINE__,group2_kernel.setArg(2, cl_sieve));
    check(__LINE__,group2_kernel.setArg(3, cl_hash_chunks));
    check(__LINE__,group2_kernel.setArg(4, cl_hash_chunks_1));
    check(__LINE__,group2_kernel.setArg(5, cl_pairs));
    check(__LINE__,group2_kernel.setArg(6, 0));
    check(__LINE__,group2_kernel.setArg(7, 0));
    check(__LINE__,group2_kernel.setArg(8, 9));
    check(__LINE__,group2_kernel.setArg(9, cl::Local(sizeof(cl_uint)*28)));
    check(__LINE__,group2_kernel.setArg(10, cl::Local(sizeof(cl_uint)*2*28)));
    check(__LINE__,queue.enqueueNDRangeKernel(group2_kernel, cl::NDRange{0}, cl::NDRange{9*28*((size+27)/28)}, cl::NDRange{28*9}));
    
    check(__LINE__,queue.finish());
    std::unique_ptr<uint32_t> hash_chunks_1 = std::unique_ptr<uint32_t>(new uint32_t[10*init_size]);
    std::unique_ptr<uint32_t> pairs = std::unique_ptr<uint32_t>(new uint32_t[2*10*init_size]);
    check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_1, CL_TRUE, 0, sizeof(cl_uint)*10*init_size, hash_chunks_1.get()));
    check(__LINE__,queue.enqueueReadBuffer(cl_pairs, CL_TRUE, 0, sizeof(cl_uint)*2*init_size, pairs.get()));
    count_non_empty = 0;
    dest_offset += 1*size;
    schedule_offset += size;
    for(size_t i = 0;i<dest_offset;i++)
    {
        uint32_t i1 = pairs.get()[2*i];
        uint32_t i2 = pairs.get()[2*i+1];
        uint32_t chunk = hash_chunks_1.get()[i*9];
        if (i1 != 0xFFFFFFFFU || i2 != 0xFFFFFFFFU) {
            assert(chunk == (hash_chunks.get()[10*i1+1]^hash_chunks.get()[10*i2+1]));
            assert(hash_chunks_1.get()[9*i+1] == (hash_chunks.get()[10*i1+2]^hash_chunks.get()[10*i2+2]));
            assert(hash_chunks_1.get()[9*i+2] == (hash_chunks.get()[10*i1+3]^hash_chunks.get()[10*i2+3]));
            assert(hash_chunks_1.get()[9*i+3] == (hash_chunks.get()[10*i1+4]^hash_chunks.get()[10*i2+4]));
            assert(hash_chunks_1.get()[9*i+4] == (hash_chunks.get()[10*i1+5]^hash_chunks.get()[10*i2+5]));
            assert(hash_chunks_1.get()[9*i+5] == (hash_chunks.get()[10*i1+6]^hash_chunks.get()[10*i2+6]));
            assert(hash_chunks_1.get()[9*i+6] == (hash_chunks.get()[10*i1+7]^hash_chunks.get()[10*i2+7]));
            assert(hash_chunks_1.get()[9*i+7] == (hash_chunks.get()[10*i1+8]^hash_chunks.get()[10*i2+8]));
            assert(hash_chunks_1.get()[9*i+8] == (hash_chunks.get()[10*i1+9]^hash_chunks.get()[10*i2+9]));
            count_non_empty++;
        }
    }
    std::cout << "Pairs in the hash_chunks_1 after group2: " << count_non_empty << "\n";
    
    size = group_offsets.get()[1];
    cl::Buffer cl_debug(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*init_size);
    std::unique_ptr<uint32_t> debug = std::unique_ptr<uint32_t>(new uint32_t[init_size]);
    check(__LINE__,group3_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group3_kernel.setArg(1, schedule_offset));
    check(__LINE__,group3_kernel.setArg(2, size));
    check(__LINE__,group3_kernel.setArg(3, cl_sieve));
    check(__LINE__,group3_kernel.setArg(4, cl_hash_chunks));
    check(__LINE__,group3_kernel.setArg(5, cl_hash_chunks_1));
    check(__LINE__,group3_kernel.setArg(6, cl_pairs));
    check(__LINE__,group3_kernel.setArg(7, 0));
    check(__LINE__,group3_kernel.setArg(8, 0));
    check(__LINE__,group3_kernel.setArg(9, dest_offset));
    check(__LINE__,group3_kernel.setArg(10, 9));
    check(__LINE__,group3_kernel.setArg(11, cl::Local(sizeof(cl_uint)*9)));
    check(__LINE__,group3_kernel.setArg(12, cl::Local(sizeof(cl_uint)*9*3)));
    check(__LINE__,group3_kernel.setArg(13, cl::Local(sizeof(cl_uint)*9*3*9)));
    check(__LINE__,queue.enqueueNDRangeKernel(group3_kernel, cl::NullRange, cl::NDRange{(3*9)*9*((size+8)/9)}, cl::NDRange{3*9*9}));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_1, CL_TRUE, 0, sizeof(cl_uint)*10*init_size, hash_chunks_1.get()));
    check(__LINE__,queue.enqueueReadBuffer(cl_pairs, CL_TRUE, 0, sizeof(cl_uint)*2*init_size, pairs.get()));
    count_non_empty = 0;
    dest_offset += 3*size;
    schedule_offset += size;
    for(size_t i = 0;i<dest_offset;i++)
    {
        uint32_t i1 = pairs.get()[2*i];
        uint32_t i2 = pairs.get()[2*i+1];
        uint32_t chunk = hash_chunks_1.get()[i*9];
        if (i1 != 0xFFFFFFFFU || i2 != 0xFFFFFFFFU) {
            assert(chunk == (hash_chunks.get()[10*i1+1]^hash_chunks.get()[10*i2+1]));
            assert(hash_chunks_1.get()[9*i+1] == (hash_chunks.get()[10*i1+2]^hash_chunks.get()[10*i2+2]));
            assert(hash_chunks_1.get()[9*i+2] == (hash_chunks.get()[10*i1+3]^hash_chunks.get()[10*i2+3]));
            assert(hash_chunks_1.get()[9*i+3] == (hash_chunks.get()[10*i1+4]^hash_chunks.get()[10*i2+4]));
            assert(hash_chunks_1.get()[9*i+4] == (hash_chunks.get()[10*i1+5]^hash_chunks.get()[10*i2+5]));
            assert(hash_chunks_1.get()[9*i+5] == (hash_chunks.get()[10*i1+6]^hash_chunks.get()[10*i2+6]));
            assert(hash_chunks_1.get()[9*i+6] == (hash_chunks.get()[10*i1+7]^hash_chunks.get()[10*i2+7]));
            assert(hash_chunks_1.get()[9*i+7] == (hash_chunks.get()[10*i1+8]^hash_chunks.get()[10*i2+8]));
            assert(hash_chunks_1.get()[9*i+8] == (hash_chunks.get()[10*i1+9]^hash_chunks.get()[10*i2+9]));
            count_non_empty++;
        }
    }
    std::cout << "Pairs in the hash_chunks_1 after group3: " << count_non_empty << "\n";
    
    size = group_offsets.get()[2];
    check(__LINE__,group4_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group4_kernel.setArg(1, schedule_offset));
    check(__LINE__,group4_kernel.setArg(2, size));
    check(__LINE__,group4_kernel.setArg(3, cl_sieve));
    check(__LINE__,group4_kernel.setArg(4, cl_hash_chunks));
    check(__LINE__,group4_kernel.setArg(5, cl_hash_chunks_1));
    check(__LINE__,group4_kernel.setArg(6, cl_pairs));
    check(__LINE__,group4_kernel.setArg(7, 0));
    check(__LINE__,group4_kernel.setArg(8, 0));
    check(__LINE__,group4_kernel.setArg(9, dest_offset));
    check(__LINE__,group4_kernel.setArg(10, 9));
    check(__LINE__,group4_kernel.setArg(11, cl::Local(sizeof(cl_uint)*7)));
    check(__LINE__,group4_kernel.setArg(12, cl::Local(sizeof(cl_uint)*7*4)));
    check(__LINE__,group4_kernel.setArg(13, cl::Local(sizeof(cl_uint)*7*4*9)));
    check(__LINE__,queue.enqueueNDRangeKernel(group4_kernel, cl::NullRange, cl::NDRange{(4*9)*7*((size+6)/7)}, cl::NDRange{4*7*9}));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_1, CL_TRUE, 0, sizeof(cl_uint)*10*init_size, hash_chunks_1.get()));
    check(__LINE__,queue.enqueueReadBuffer(cl_pairs, CL_TRUE, 0, sizeof(cl_uint)*2*init_size, pairs.get()));
    count_non_empty = 0;
    dest_offset += 6*size;
    schedule_offset += size;
    for(size_t i = 0;i<dest_offset;i++)
    {
        uint32_t i1 = pairs.get()[2*i];
        uint32_t i2 = pairs.get()[2*i+1];
        uint32_t chunk = hash_chunks_1.get()[i*9];
        if (i1 != 0xFFFFFFFFU || i2 != 0xFFFFFFFFU) {
            assert(chunk == (hash_chunks.get()[10*i1+1]^hash_chunks.get()[10*i2+1]));
            assert(hash_chunks_1.get()[9*i+1] == (hash_chunks.get()[10*i1+2]^hash_chunks.get()[10*i2+2]));
            assert(hash_chunks_1.get()[9*i+2] == (hash_chunks.get()[10*i1+3]^hash_chunks.get()[10*i2+3]));
            assert(hash_chunks_1.get()[9*i+3] == (hash_chunks.get()[10*i1+4]^hash_chunks.get()[10*i2+4]));
            assert(hash_chunks_1.get()[9*i+4] == (hash_chunks.get()[10*i1+5]^hash_chunks.get()[10*i2+5]));
            assert(hash_chunks_1.get()[9*i+5] == (hash_chunks.get()[10*i1+6]^hash_chunks.get()[10*i2+6]));
            assert(hash_chunks_1.get()[9*i+6] == (hash_chunks.get()[10*i1+7]^hash_chunks.get()[10*i2+7]));
            assert(hash_chunks_1.get()[9*i+7] == (hash_chunks.get()[10*i1+8]^hash_chunks.get()[10*i2+8]));
            assert(hash_chunks_1.get()[9*i+8] == (hash_chunks.get()[10*i1+9]^hash_chunks.get()[10*i2+9]));
            count_non_empty++;
        }
    }
    std::cout << "Pairs in the hash_chunks_1 after group4: " << count_non_empty << "\n";
    
    size = group_offsets.get()[3];
    check(__LINE__,group5_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,group5_kernel.setArg(1, schedule_offset));
    check(__LINE__,group5_kernel.setArg(2, size));
    check(__LINE__,group5_kernel.setArg(3, cl_sieve));
    check(__LINE__,group5_kernel.setArg(4, cl_hash_chunks));
    check(__LINE__,group5_kernel.setArg(5, cl_hash_chunks_1));
    check(__LINE__,group5_kernel.setArg(6, cl_pairs));
    check(__LINE__,group5_kernel.setArg(7, 0));
    check(__LINE__,group5_kernel.setArg(8, 0));
    check(__LINE__,group5_kernel.setArg(9, dest_offset));
    check(__LINE__,group5_kernel.setArg(10, 9));
    check(__LINE__,group5_kernel.setArg(11, cl::Local(sizeof(cl_uint)*5)));
    check(__LINE__,group5_kernel.setArg(12, cl::Local(sizeof(cl_uint)*5*5)));
    check(__LINE__,group5_kernel.setArg(13, cl::Local(sizeof(cl_uint)*5*5*9)));
    check(__LINE__,queue.enqueueNDRangeKernel(group5_kernel, cl::NullRange, cl::NDRange{(5*9)*5*((size+4)/5)}, cl::NDRange{5*5*9}));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_1, CL_TRUE, 0, sizeof(cl_uint)*10*init_size, hash_chunks_1.get()));
    check(__LINE__,queue.enqueueReadBuffer(cl_pairs, CL_TRUE, 0, sizeof(cl_uint)*2*init_size, pairs.get()));
    count_non_empty = 0;
    dest_offset += 10*size;
    schedule_offset += size;
    for(size_t i = 0;i<dest_offset;i++)
    {
        uint32_t i1 = pairs.get()[2*i];
        uint32_t i2 = pairs.get()[2*i+1];
        uint32_t chunk = hash_chunks_1.get()[i*9];
        if (i1 != 0xFFFFFFFFU || i2 != 0xFFFFFFFFU) {
            assert(chunk == (hash_chunks.get()[10*i1+1]^hash_chunks.get()[10*i2+1]));
            assert(hash_chunks_1.get()[9*i+1] == (hash_chunks.get()[10*i1+2]^hash_chunks.get()[10*i2+2]));
            assert(hash_chunks_1.get()[9*i+2] == (hash_chunks.get()[10*i1+3]^hash_chunks.get()[10*i2+3]));
            assert(hash_chunks_1.get()[9*i+3] == (hash_chunks.get()[10*i1+4]^hash_chunks.get()[10*i2+4]));
            assert(hash_chunks_1.get()[9*i+4] == (hash_chunks.get()[10*i1+5]^hash_chunks.get()[10*i2+5]));
            assert(hash_chunks_1.get()[9*i+5] == (hash_chunks.get()[10*i1+6]^hash_chunks.get()[10*i2+6]));
            assert(hash_chunks_1.get()[9*i+6] == (hash_chunks.get()[10*i1+7]^hash_chunks.get()[10*i2+7]));
            assert(hash_chunks_1.get()[9*i+7] == (hash_chunks.get()[10*i1+8]^hash_chunks.get()[10*i2+8]));
            assert(hash_chunks_1.get()[9*i+8] == (hash_chunks.get()[10*i1+9]^hash_chunks.get()[10*i2+9]));
            count_non_empty++;
        }
    }
    std::cout << "Pairs in the hash_chunks_1 after group5: " << count_non_empty << "\n";
    
    size = group_offsets.get()[4];
    check(__LINE__,groupc_kernel.setArg(0, cl_schedule_v2));
    check(__LINE__,groupc_kernel.setArg(1, schedule_offset));
    check(__LINE__,groupc_kernel.setArg(2, size));
    check(__LINE__,groupc_kernel.setArg(3, cl_sieve));
    check(__LINE__,groupc_kernel.setArg(4, cl_hash_chunks));
    check(__LINE__,groupc_kernel.setArg(5, cl_hash_chunks_1));
    check(__LINE__,groupc_kernel.setArg(6, cl_pairs));
    check(__LINE__,groupc_kernel.setArg(7, 0));
    check(__LINE__,groupc_kernel.setArg(8, 0));
    check(__LINE__,groupc_kernel.setArg(9, dest_offset));
    check(__LINE__,groupc_kernel.setArg(10, (cl_uchar)9));
    check(__LINE__,groupc_kernel.setArg(11, (cl_uchar)6));
    check(__LINE__,groupc_kernel.setArg(12, cl::Local(sizeof(cl_uint)*4)));
    check(__LINE__,groupc_kernel.setArg(13, cl::Local(sizeof(cl_uint)*4*6)));
    check(__LINE__,groupc_kernel.setArg(14, cl::Local(sizeof(cl_uint)*4*6*9)));
    check(__LINE__,queue.enqueueNDRangeKernel(groupc_kernel, cl::NullRange, cl::NDRange{(6*9)*4*((size+3)/4)}, cl::NDRange{4*6*9}));
    check(__LINE__,queue.finish());
    check(__LINE__,queue.enqueueReadBuffer(cl_hash_chunks_1, CL_TRUE, 0, sizeof(cl_uint)*10*init_size, hash_chunks_1.get()));
    check(__LINE__,queue.enqueueReadBuffer(cl_pairs, CL_TRUE, 0, sizeof(cl_uint)*2*init_size, pairs.get()));
    count_non_empty = 0;
    dest_offset += 15*size;
    schedule_offset += size;
    for(size_t i = 0;i<dest_offset;i++)
    {
        uint32_t i1 = pairs.get()[2*i];
        uint32_t i2 = pairs.get()[2*i+1];
        uint32_t chunk = hash_chunks_1.get()[i*9];
        if (i1 != 0xFFFFFFFFU || i2 != 0xFFFFFFFFU) {
            assert(chunk == (hash_chunks.get()[10*i1+1]^hash_chunks.get()[10*i2+1]));
            assert(hash_chunks_1.get()[9*i+1] == (hash_chunks.get()[10*i1+2]^hash_chunks.get()[10*i2+2]));
            assert(hash_chunks_1.get()[9*i+2] == (hash_chunks.get()[10*i1+3]^hash_chunks.get()[10*i2+3]));
            assert(hash_chunks_1.get()[9*i+3] == (hash_chunks.get()[10*i1+4]^hash_chunks.get()[10*i2+4]));
            assert(hash_chunks_1.get()[9*i+4] == (hash_chunks.get()[10*i1+5]^hash_chunks.get()[10*i2+5]));
            assert(hash_chunks_1.get()[9*i+5] == (hash_chunks.get()[10*i1+6]^hash_chunks.get()[10*i2+6]));
            assert(hash_chunks_1.get()[9*i+6] == (hash_chunks.get()[10*i1+7]^hash_chunks.get()[10*i2+7]));
            assert(hash_chunks_1.get()[9*i+7] == (hash_chunks.get()[10*i1+8]^hash_chunks.get()[10*i2+8]));
            assert(hash_chunks_1.get()[9*i+8] == (hash_chunks.get()[10*i1+9]^hash_chunks.get()[10*i2+9]));
            count_non_empty++;
        }
    }
    std::cout << "Pairs in the hash_chunks_1 after group6: " << count_non_empty << "\n";
    
    /*
     for(size_t i = 0;i<32;i++)
     {
     const uint32_t item = schedule.get()[i];
     std::cout << i << ":[" << (item>>8) << "|" << (item&0xFF) << "] ";
     }
     std::cout << "\n";
     */
    cl::Event blake_event;
    cl::Event prepare_event;
    std::vector<cl::Event> clear_sieve_events;
    std::vector<cl::Event> fill_sieve_events;
    std::vector<cl::Event> correct_sieve_events;
    std::vector<cl::Event> align_events;
    std::vector<cl::Event> subarray_events;
    std::vector<cl::Event> subarray_inc_events;
    std::vector<cl::Event> clear_schedule_events;
    std::vector<cl::Event> project_events;
    std::vector<cl::Event> smear_events;
    std::vector<cl::Event> make_pairs_events;
    std::vector<cl::Event> xor_events;
    std::vector<cl::Event> xorv2_events;
    std::vector<cl::Event> xor_copy_events;
    std::vector<cl::Event> xor_xor_events;
    std::vector<cl::Event> group_count_events;
    std::vector<cl::Event> group_offsets_events;
    std::vector<cl::Event> group2_events;
    std::vector<cl::Event> group3_events;
    std::vector<cl::Event> group4_events;
    std::vector<cl::Event> group5_events;
    std::vector<cl::Event> groupc_events;
    cl::Event find_zeros_event;
    cl::Event get_indices_event;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    check(__LINE__,queue.enqueueWriteBuffer(cl_blake, CL_TRUE, 0, sizeof(blake.h), blake.h));
    check(__LINE__,blake_kernel.setArg(0, cl_blake));
    check(__LINE__,blake_kernel.setArg(1, cl_hashes));
    check(__LINE__,queue.enqueueNDRangeKernel(blake_kernel, cl::NullRange,cl::NDRange{init_size/2},cl::NDRange{64}, nullptr, &blake_event));
    check(__LINE__,prepare_kernel.setArg(0, cl_hashes));
    check(__LINE__,prepare_kernel.setArg(1, cl_hash_chunks));
    check(__LINE__,queue.enqueueNDRangeKernel(prepare_kernel, cl::NullRange,cl::NDRange{init_size/2, 20},cl::NDRange{1, 20}, nullptr, &prepare_event));
    uint problem_size = init_size;
    cl_uint pair_offset = 0;
    cl_uint pair_prev_offset = 0;
    for (cl_uint i=9; i>=2; --i) {
        std::cout << "Problem size =" << problem_size << "\n";
        problem_size = solution_cycle_v2(queue, (i%2==0)?cl_hash_chunks_1:cl_hash_chunks, cl_sieve, cl_pair_counts, cl_group_partial_one, cl_group_partial_two, cl_pairs, (i%2==0)?cl_hash_chunks:cl_hash_chunks_1, cl_schedule_v2, cl_group_counts, cl_group_offsets, group_offsets.get(), clear_sieve_kernel, clear_sieve_events, fill_sieve_kernel, fill_sieve_events, correct_sieve_kernel, correct_sieve_events, align_collisions_kernel, align_events, group_count_kernel, group_count_events, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events, group_offsets_kernel, group_offsets_events, clear_schedule_v2, clear_schedule_events, project_v2, project_events, group2_kernel, group2_events, group3_kernel, group3_events, group4_kernel, group4_events, group5_kernel, group5_events, groupc_kernel, groupc_events, KeysCount, problem_size, max_workgroup_size, i, pair_prev_offset, pair_offset, false);
        pair_prev_offset = pair_offset;
        pair_offset += problem_size;
    }
    problem_size = solution_cycle_final(queue, cl_hash_chunks, cl_sieve, cl_pair_counts, cl_group_partial_one, cl_group_partial_two, cl_pairs, cl_schedule_v2, cl_group_counts, cl_group_offsets, group_offsets.get(), clear_sieve_kernel, clear_sieve_events, fill_sieve_kernel, fill_sieve_events, correct_sieve_kernel, correct_sieve_events, align_collisions_kernel, align_events, group_count_kernel, group_count_events, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events, group_offsets_kernel, group_offsets_events, clear_schedule_v2, clear_schedule_events, project_v2, project_events, final2_kernel, group2_events, final3_kernel, group3_events,final4_kernel, group4_events, final5_kernel, group5_events, finalc_kernel, groupc_events, KeysCount, problem_size, max_workgroup_size, pair_prev_offset, pair_offset);
    
    
    check(__LINE__,find_zeros_kernel.setArg(0, cl_pairs));
    check(__LINE__,find_zeros_kernel.setArg(1, pair_offset));
    check(__LINE__,find_zeros_kernel.setArg(2, cl_zero_counts));
    check(__LINE__,queue.enqueueNDRangeKernel(find_zeros_kernel, cl::NullRange, cl::NDRange{problem_size}, cl::NullRange, nullptr, &find_zeros_event));
    
    prefix_sum(queue, cl_zero_counts, problem_size + 1, max_workgroup_size, cl_partzero_one, cl_partzero_two, scan_subarrays_kernel, subarray_events, scan_inc_subarrays_kernel, subarray_inc_events);
    
    check(__LINE__,get_indices_kernel.setArg(0, cl_zero_counts));
    check(__LINE__,get_indices_kernel.setArg(1, problem_size));
    check(__LINE__,get_indices_kernel.setArg(2, cl_pairs));
    check(__LINE__,get_indices_kernel.setArg(3, pair_offset));
    check(__LINE__,get_indices_kernel.setArg(4, cl_indices));
    check(__LINE__,queue.enqueueNDRangeKernel(get_indices_kernel, cl::NullRange, cl::NDRange{problem_size,256}, cl::NDRange{1,256}, nullptr, &get_indices_event));
    check(__LINE__,queue.finish());
    
    std::unique_ptr<uint32_t> zero_counts = std::unique_ptr<uint32_t>(new uint32_t[problem_size+1]);
    check(__LINE__,queue.enqueueReadBuffer(cl_zero_counts, CL_TRUE, 0, sizeof(cl_uint)*(problem_size+1), zero_counts.get()));
    //std::cout << "Zeros after scan: " << zero_counts.get()[problem_size] << "\n";
    std::unique_ptr<uint32_t> indices = std::unique_ptr<uint32_t>(new uint32_t[256*512]);
    check(__LINE__,queue.enqueueReadBuffer(cl_indices, CL_TRUE, 0, sizeof(cl_uint)*256*512, indices.get()));
    size_t zeros = zero_counts.get()[problem_size];
    if (zeros > 256) {
        zeros = 256;
    }
    size_t distinct_sets = 0;
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
            //assert(HasCollision(X[i1], X[i2], CollisionByteLength));
            Y1.emplace_back(X[i1], X[i2], 30, 4, 3);
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
        //assert(HasCollision(Y8[0], Y8[1], CollisionByteLength*2));
        if (DistinctIndices(Y8[0], Y8[1], 6, 1024)) {
            FullStepRow<2*FullWidth> res(Y8[0], Y8[1], 6, 1024, 0);
            auto soln = res.GetIndices(6, 2*1024, CollisionBitLength);
            assert(soln.size() == equihash_solution_size(N, K));
            if (validBlock(soln)) {
                return true;
            }
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "9 rounds in us = " << std::dec << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() <<std::endl;
    std::cout << "Distinct sets: " << distinct_sets<< "\n";
    cl_ulong blake_time = blake_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - blake_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Blake took " << blake_time << " ns\n";
    cl_ulong prepare_time = prepare_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prepare_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Prepare took " << prepare_time << " ns\n";
    std::cout << "Clear sieve took ";
    for(const auto& event: clear_sieve_events) {
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
    std::cout << "Correct sieve took ";
    for(const auto& event: correct_sieve_events) {
        cl_ulong time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::cout << time << " ";
    }
    std::cout << " ns\n";
    std::cout << "Align took ";
    for(const auto& event: align_events) {
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
    std::cout << "Clear schedule took ";
    for(const auto& event: clear_schedule_events) {
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
    cl_ulong get_indices_time = get_indices_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - get_indices_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Get indices took " << get_indices_time << " ns\n";
    return false;
}
