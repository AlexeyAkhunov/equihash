//
//  kernels.hpp
//  FXTrue
//
//  Created by Alexey Akhunov on 28/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#ifndef kernels_h
#define kernels_h

cl::Kernel& kernelByName(std::vector<cl::Kernel>& kernels, const std::string& name)
{
    for(std::size_t i = 0; i < kernels.size(); i++) {
        std::string n;
        kernels[i].getInfo(CL_KERNEL_FUNCTION_NAME, &n);
        std::string name_1(name + (char)0);
        if (name == n || name_1 == n)
        {
            return kernels[i];
        }
    }
    throw std::invalid_argument("Could not find kernel with name " + name);
}

std::string kernel_source = R"(

#define N 200
#define K 9
#define INDICES_PER_HASH (512/N)
#define RAW_ROW_SIZE_BYTES (N*INDICES_PER_HASH/8)
#define RAW_ROW_SIZE_WORDS ((RAW_ROW_SIZE_BYTES+3)/4)
#define CHUNKS (K+1)
#define BITS_IN_CHUNK (N/CHUNKS)
#define WG_SIZE (INDICES_PER_HASH*CHUNKS)
#define NR_INPUTS (1 << BITS_IN_CHUNK)

#define mix(va, vb, vc, vd, x, y) \
va = (va + vb + x); \
vd = rotate((vd ^ va), (ulong)64 - 32); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (ulong)64 - 24); \
va = (va + vb + y); \
vd = rotate((vd ^ va), (ulong)64 - 16); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (ulong)64 - 63);

__constant ulong blake_iv[] =
{
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void compute_blake(global ulong *blake_state, global ulong *hashes)
{
    uint                tid = get_global_id(0);
    ulong               v[16];
    uint                inputs_per_thread = NR_INPUTS / get_global_size(0);
    uint                input = tid * inputs_per_thread;
    uint                input_end = (tid + 1) * inputs_per_thread;
    while (input < input_end)
    {
        // shift "i" to occupy the high 32 bits of the second ulong word in the
        // message block
        ulong word1 = (ulong)input << 32;
        // init vector v
        v[0] = blake_state[0];
        v[1] = blake_state[1];
        v[2] = blake_state[2];
        v[3] = blake_state[3];
        v[4] = blake_state[4];
        v[5] = blake_state[5];
        v[6] = blake_state[6];
        v[7] = blake_state[7];
        v[8] =  blake_iv[0];
        v[9] =  blake_iv[1];
        v[10] = blake_iv[2];
        v[11] = blake_iv[3];
        v[12] = blake_iv[4];
        v[13] = blake_iv[5];
        v[14] = blake_iv[6];
        v[15] = blake_iv[7];
        // mix in length of data
        v[12] ^= 140 + 4 /* length of "i" */;
        // last block
        v[14] ^= -1;
        
        // round 1
        mix(v[0], v[4], v[8],  v[12], 0, word1);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 2
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], word1, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 3
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, word1);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 4
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, word1);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 5
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, word1);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 6
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], word1, 0);
        // round 7
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], word1, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 8
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, word1);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 9
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], word1, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 10
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], word1, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 11
        mix(v[0], v[4], v[8],  v[12], 0, word1);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 12
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], word1, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        
        // compress v into the blake state; this produces the 50-byte hash
        // (two Xi values)
        ulong h[8];
        h[0] = blake_state[0] ^ v[0] ^ v[8];
        h[1] = blake_state[1] ^ v[1] ^ v[9];
        h[2] = blake_state[2] ^ v[2] ^ v[10];
        h[3] = blake_state[3] ^ v[3] ^ v[11];
        h[4] = blake_state[4] ^ v[4] ^ v[12];
        h[5] = blake_state[5] ^ v[5] ^ v[13];
        h[6] = blake_state[6] ^ v[6] ^ v[14];
        //h[7] = blake_state[7] ^ v[7] ^ v[15];
        
        hashes[8*input] = h[0];
        hashes[8*input+1] = h[1];
        hashes[8*input+2] = h[2];
        hashes[8*input+3] = h[3];
        hashes[8*input+4] = h[4];
        hashes[8*input+5] = h[5];
        hashes[8*input+6] = h[6];
        //hashes[8*input+7] = h[7];
        
        input++;
    }

}

// Splits hashes that contain N bits into the chunks that contain N/(K+1) bits
// There are 2**(N/(K+1)+1) hashes in the input
// This kernel operates in two dimensions:
// 1. Rows - 2**(N/(K+1)+1)
// 2. Within each row, chunk - there are K+1 chunks
// Each work group first loads (N+31)/32 words (uints) into the local (shared) memory
// and then splits it up in N/(K+1) chunks
__attribute__((reqd_work_group_size(1, WG_SIZE, 1)))
kernel void prepare(const global uint* hashes, global uint* hash_chunks, global uint* sieve, global uint* pair_counts)
{
    const uint row = get_global_id(0);
    const uint chunk = get_global_id(1);
    local uint raw[RAW_ROW_SIZE_WORDS];
    // Load the data
    if (chunk < RAW_ROW_SIZE_WORDS)
    {
        uint x = hashes[16*row+chunk];
        raw[chunk] = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4); // Flip nibbles in the bytes
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Split into the chunks
    const size_t start_bit = chunk * BITS_IN_CHUNK;
    // Assuming that the GPU is big-endian
    const uint v = ((raw[start_bit>>5] >> (start_bit&31)) | (raw[(start_bit + 20)>>5] << (32-(start_bit&31))))&0xfffff;
    if (chunk%10==0)
    {
        const uint col = atomic_inc(pair_counts + v);
        if (col < 16)
        {
            sieve[16*v+col] = 2*row + (chunk/10);
        }
    }
    else
    {
        hash_chunks[20*row+chunk] = v;
    }
}

kernel void clear_counts(global uint* pair_counts)
{
    pair_counts[get_global_id(0)] = 0;
}

kernel void fill_sieve_atomic(const global uint* vchunks, global uint* sieve, global uint* pair_counts)
{
    const uint gid = get_global_id(0);
    const uint row = vchunks[gid];
    const uint col = atomic_inc(pair_counts + row);
    if (col < 16)
    {
        sieve[16*row+col] = gid;
    }
}

kernel void group_count_v2(const global uint* pair_counts, global uint* group_counts)
{
}

kernel void group_count(const global uint* pair_counts,
                        global uint* group_counts)
{
    local uint count[16];
    const uint gid = get_global_id(0);
    const uint size = get_global_size(0);
    const uint lid = get_local_id(0);
    const uint c = get_local_id(1);
    if (c == 0)
    {
        const uint x = pair_counts[gid];
        count[lid] = x > 16 ? 16 : x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    group_counts[c*size+gid] = count[lid] == (c+2);
}

/*
 * First phase of a multiblock scan.
 *
 * Given a global array [data] of length arbitrary length [n].
 * We assume that we have k workgroups each of size m/2 workitems.
 * Each workgroup handles a subarray of length [m] (where m is a power of two).
 * The last subarray will be padded with 0 if necessary (n < k*m).
 * We use the primitives above to perform a scan operation within each subarray.
 * We store the intermediate reduction of each subarray (following upsweep_pow2) in [part].
 * These partial values can themselves be scanned and fed into [scan_inc_subarrays].
 */
kernel void scan_subarrays(
                           local uint* x,
                           global uint *data, //length [n]
                           global uint *part, //length [m]
                           uint n
                           ) {
    // workgroup size
    const uint wx = 256;
    // global identifiers and indexes
    uint gid = get_global_id(0);
    // local identifiers and indexes
    uint lid = get_local_id(0);
    
    // copy into local data padding elements >= n with 0
    x[lid+(lid>>4)] = (2*gid-lid < n-1) ? data[2*gid-lid] : 0;
    x[wx+lid+((wx+lid)>>4)] = (2*gid+wx-lid < n-1) ? data[2*gid+wx-lid] : 0;
    
    int offset = 1;
#pragma unroll
    for (uint d = 256; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            int ai = offset*(2*lid+1)-1;
            ai += (ai>>4);
            int bi = offset*(2*lid+2)-1;
            bi += (bi>>4);
            x[bi] += x[ai];
        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int last = 2*wx-1;
        last += (last>>4);
        if (n > 2*wx) {
            part[get_group_id(0)] = x[last];
        }
        x[last] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (uint d = 1; d <= 256; d *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            int ai = offset*(2*lid+1)-1;
            ai += (ai>>4);
            int bi = offset*(2*lid+2)-1;
            bi += (bi>>4);
            uint t = x[ai];
            x[ai] = x[bi];
            x[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // copy back to global data
    if (2*gid-lid < n) {
        data[2*gid-lid] = x[lid+(lid>>4)];
    }
    if ((2*gid+wx-lid) < n) {
        data[2*gid+wx-lid] = x[wx+lid+((wx+lid)>>4)];
    }
    
}

/*
 * Perform the second phase of an inplace exclusive scan on a global array [data] of arbitrary length [n].
 *
 * We assume that we have k workgroups each of size m/2 workitems.
 * Each workgroup handles a subarray of length [m] (where m is a power of two).
 * We sum each element by the sum of the preceding subarrays taken from [part].
 */
kernel void scan_inc_subarrays(
                               local uint* x,
                               global uint *data, //length [n]
                               global uint *part, //length [m]
                               uint n
                               ) {
    // global identifiers and indexes
    uint gid = get_global_id(0);
    // local identifiers and indexes
    uint lid = get_local_id(0);
    uint wx = get_local_size(0);
    local uint my_part;
    
    if (lid == 0 && n > 2*get_local_size(0)) {
        my_part = part[get_group_id(0)];
    }
    
    // copy into local data padding elements >= n with identity
    x[lid] = (2*gid-lid < n) ? data[2*gid-lid] : 0;
    x[wx+lid] = (2*gid+wx-lid < n) ? data[2*gid+wx-lid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (n > 2*get_local_size(0)) {
        x[2*lid] += my_part;
        x[2*lid+1] += my_part;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // copy back to global data
    if (2*gid-lid < n) {
        data[2*gid-lid] = x[lid];
    }
    if (2*gid+wx-lid < n) {
        data[2*gid+wx-lid] = x[wx+lid];
    }
}

kernel void project(const global uint* pair_counts,
                       const global uint* group_counts,
                       global uint* schedule)
{
    const uint gid = get_global_id(0);
    const uint size = get_global_size(0);
    uint count = pair_counts[gid];
    if (count < 2)
    {
        return;
    }
    if (count > 16)
    {
        count = 16;
    }
    const uint idx = group_counts[(count-2)*size+gid];
    schedule[idx] = gid;
}

kernel void group_offsets(const global uint* group_counts,
                          global uint* group_offsets,
                          const uint size)
{
    const uint gid = get_global_id(0);
    group_offsets[gid] = group_counts[size*(gid+1)] - group_counts[size*gid];
}

kernel void find_zeros(const global uint* pairs, const uint pair_offset, global uint* zero_counts)
{
    uint gid = get_global_id(0);
    zero_counts[gid] = (pairs[2*(pair_offset+gid)] != 0);
}

kernel void project_zeros(const global uint* zero_counts, global uint* schedule)
{
    const uint gid = get_global_id(0);
    uint zero_count = zero_counts[gid];
    if (zero_count != zero_counts[gid+1])
    {
        schedule[zero_count] = gid;
    }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void get_indices(const global uint* schedule,
                        const global uint* pairs,
                        const uint pair_offset,
                        global uint* indices)
{
    local uint x[256];
    uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint idx = pair_offset + schedule[gid];
    if (lid < 2)
    {
        x[lid] = pairs[2*idx + lid];
    }
    for(uint d=7,t=4;d > 0;--d,t<<=1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < t)
        {
            const uint prev_index = (lid<t)?x[lid>>1]:0;
            x[lid] = pairs[2*prev_index+(lid&1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    indices[gid*512+lid] = pairs[2*x[lid/2]+(lid%2)];
    indices[gid*512+256+lid] = pairs[2*x[128+(lid/2)]+(lid%2)];
}

kernel void group2(
                   const global uint* schedule,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* hash_chunks_dst,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint num_chunks,
                   local uint* sieve_index,
                   local uint* indices,
                   global uint* vchunks
                   )
{
    const ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/num_chunks;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[grp_offset+lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 2*num_groups)
    {
        indices[lid] = sieve[16*sieve_index[lid/2]+(lid%2)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 2*num_groups && (grp_offset+(lid/2)) < n)
    {
        pairs[2*(pairs_offset+grp_offset)+lid] = pairs_prev_offset + indices[lid];
    }
    if (grp_offset+(lid/num_chunks) < n)
    {
        const uint v = hash_chunks_src[(num_chunks+1)*(indices[2*(lid/num_chunks)])+(lid%num_chunks)+1]^hash_chunks_src[(num_chunks+1)*(indices[2*(lid/num_chunks)+1])+(lid%num_chunks)+1];
        if (lid%num_chunks == 0)
        {
            vchunks[grp_offset+(lid/num_chunks)] = v;
        }
        else
        {
            hash_chunks_dst[num_chunks*(grp_offset+(lid/num_chunks))+(lid%num_chunks)] = v;
        }
    }
}

kernel void final2(
                   const global uint* schedule,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   local uint* sieve_index,
                   local uint* indices
                   )
{
    const ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0);
    const uint grp_offset = get_group_id(0)*num_groups;
    sieve_index[lid] = schedule[grp_offset+lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    indices[2*lid] = sieve[16*sieve_index[lid]];
    indices[2*lid+1] = sieve[16*sieve_index[lid]+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (grp_offset+lid < n)
    {
        const bool equal = hash_chunks_src[2*(indices[2*lid])+1] == hash_chunks_src[2*(indices[2*lid+1])+1];
        pairs[2*(pairs_offset+grp_offset+lid)] = equal ? (pairs_prev_offset + indices[2*lid]) : 0;
        pairs[2*(pairs_offset+grp_offset+lid)+1] = equal ? (pairs_prev_offset + indices[2*lid+1]) : 0;
    }
}

kernel void group3(
                   const global uint* schedule,
                   const uint schedule_offset,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* hash_chunks_dst,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint dest_offset,
                   const uint num_chunks,
                   local uint* sieve_index,
                   local uint* indices,
                   local uint* chunks,
                   global uint* vchunks
                   )
{
    const ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/num_chunks/3;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 3*num_groups)
    {
        indices[lid] = sieve[16*sieve_index[lid/3]+(lid%3)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((lid < 2*3*num_groups) && (grp_offset+((lid/2)/3) < n))
    {
        const ushort prefix_idx = 3*((lid/2)/3);
        const uchar idx = (lid/2)%3; // 0..2
        const uchar next_idx = (idx+(lid%2))%3; // 0..2
        pairs[2*(pairs_offset+dest_offset+3*grp_offset)+lid] = pairs_prev_offset + indices[prefix_idx+next_idx];
    }
    chunks[lid] = hash_chunks_src[(num_chunks+1)*indices[lid/num_chunks]+(lid%num_chunks)+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (grp_offset+(lid/(3*num_chunks)) < n)
    {
        const ushort prefix_idx = 3*num_chunks*(lid/(3*num_chunks));
        const uchar idx = (lid%(3*num_chunks))/num_chunks; // 0..2
        const uchar next_idx = (idx + 1)%3; // 0..2
        const uint v = chunks[lid]^chunks[prefix_idx+next_idx*num_chunks+(lid%num_chunks)];
        if (lid%num_chunks==0)
        {
            vchunks[dest_offset+3*grp_offset+(lid/num_chunks)] = v;
        }
        else
        {
            hash_chunks_dst[num_chunks*(dest_offset+3*grp_offset)+lid] = v;
        }
    }
}

kernel void final3(
                   const global uint* schedule,
                   const uint schedule_offset,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint dest_offset,
                   local uint* sieve_index,
                   local uint* indices,
                   local uint* chunks
                   )
{
    const ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/3;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    indices[lid] = sieve[16*sieve_index[lid/3]+(lid%3)];
    barrier(CLK_LOCAL_MEM_FENCE);
    chunks[lid] = hash_chunks_src[2*indices[lid]+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (grp_offset+(lid/3) < n)
    {
        const ushort prefix_idx = 3*(lid/3);
        const uchar next_idx = ((lid%3)+1)%3; // 0..2
        const bool equal = chunks[lid] == chunks[prefix_idx+next_idx];
        pairs[2*(pairs_offset+dest_offset+3*grp_offset+lid)] = equal ? (pairs_prev_offset + indices[lid]) : 0;
        pairs[2*(pairs_offset+dest_offset+3*grp_offset+lid)+1] = equal ? (pairs_prev_offset + indices[prefix_idx+next_idx]) : 0;
    }
}

kernel void group4(
                   const global uint* schedule,
                   const uint schedule_offset,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* hash_chunks_dst,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint dest_offset,
                   const uint num_chunks,
                   local uint* sieve_index,
                   local uint* indices,
                   local uint* chunks,
                   global uint* vchunks
                   )
{
    const ushort num_groups = get_local_size(0)/num_chunks/4;
    ushort lid = get_local_id(0);
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 4*num_groups)
    {
        indices[lid] = sieve[16*sieve_index[lid/4]+(lid%4)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    chunks[lid] = hash_chunks_src[(num_chunks+1)*indices[lid/num_chunks]+(lid%num_chunks)+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    while (lid < 6*num_groups*num_chunks)
    {
        if (grp_offset+(lid/(6*num_chunks)) < n)
        {
            const ushort prefix_idx = 4*num_chunks*(lid/(6*num_chunks));
            const uchar idx = (lid%(6*num_chunks))/num_chunks; // 0..5
            const uchar my_idx = idx%4; // 0..3
            const uchar next_idx = (my_idx + 1 + (idx/4))%4; // 0..3
            const uint v = chunks[prefix_idx+my_idx*num_chunks+(lid%num_chunks)]^chunks[prefix_idx+next_idx*num_chunks+(lid%num_chunks)];
            if (lid%num_chunks == 0)
            {
                vchunks[dest_offset+6*grp_offset+(lid/num_chunks)] = v;
            }
            else
            {
                hash_chunks_dst[num_chunks*(dest_offset+6*grp_offset)+lid] = v;
            }
            
        }
        if ((lid < 2*6*num_groups) && (grp_offset+((lid/2)/6) < n))
        {
            const ushort prefix_idx = 4*((lid/2)/6);
            const uchar idx = (lid/2)%6; // 0..5
            const uchar next_idx = ((idx%4)+(1+(idx/4))*(lid%2))%4; // 0..3
            pairs[2*(pairs_offset+dest_offset+6*grp_offset)+lid] = pairs_prev_offset + indices[prefix_idx+next_idx];
        }
        lid += 4*num_groups*num_chunks;
    }
}

kernel void final4(
                   const global uint* schedule,
                   const uint schedule_offset,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint dest_offset,
                   local uint* sieve_index,
                   local uint* indices,
                   local uint* chunks
                   )
{
    const ushort num_groups = get_local_size(0)/4;
    ushort lid = get_local_id(0);
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    indices[lid] = sieve[16*sieve_index[lid/4]+(lid%4)];
    barrier(CLK_LOCAL_MEM_FENCE);
    chunks[lid] = hash_chunks_src[2*indices[lid]+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    while (lid < 6*num_groups)
    {
        if (grp_offset+(lid/6) < n)
        {
            const ushort prefix_idx = 4*(lid/6);
            const uchar idx = lid%6; // 0..5
            const uchar my_idx = idx%4;
            const uchar next_idx = (my_idx + 1 + (idx/4))%4; // 0..3
            const uchar equal = chunks[prefix_idx+my_idx] == chunks[prefix_idx+next_idx];
            pairs[2*(pairs_offset+dest_offset+6*grp_offset+lid)] = equal ? (pairs_prev_offset + indices[prefix_idx+my_idx]) : 0;
            pairs[2*(pairs_offset+dest_offset+6*grp_offset+lid)+1] = equal ? (pairs_prev_offset + indices[prefix_idx+next_idx]) : 0;
        }
        lid += 4*num_groups;
    }
}

kernel void group5(
                   const global uint* schedule,
                   const uint schedule_offset,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* hash_chunks_dst,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint dest_offset,
                   const uint num_chunks,
                   local uint* sieve_index,
                   local uint* indices,
                   local uint* chunks,
                   global uint* vchunks
                   )
{
    ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/num_chunks/5;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < 5*num_groups)
    {
        indices[lid] = sieve[16*sieve_index[lid/5]+(lid%5)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    chunks[lid] = hash_chunks_src[(num_chunks+1)*indices[lid/num_chunks]+(lid%num_chunks)+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    while (lid < 10*num_groups*num_chunks)
    {
        if (grp_offset+(lid/(10*num_chunks)) < n)
        {
            const ushort prefix_idx = 5*num_chunks*(lid/(10*num_chunks));
            const uchar idx = (lid%(10*num_chunks))/num_chunks; // 0..9
            const uchar my_idx = idx%5; // 0..4
            const uchar next_idx = (my_idx + 1 + (idx/5))%5; // 0..4
            const uint v = chunks[prefix_idx+my_idx*num_chunks+(lid%num_chunks)]^chunks[prefix_idx+next_idx*num_chunks+(lid%num_chunks)];
            if (lid%num_chunks == 0)
            {
                vchunks[dest_offset+10*grp_offset+(lid/num_chunks)] = v;
            }
            else
            {
                hash_chunks_dst[num_chunks*(dest_offset+10*grp_offset)+lid] = v;
            }
        }
        if ((lid < 2*10*num_groups) && (grp_offset+((lid/2)/10) < n))
        {
            const ushort prefix_idx = 5*((lid/2)/10);
            const uchar idx = (lid/2)%10; // 0..9
            const uchar next_idx = ((idx%5)+(1+(idx/5))*(lid%2))%5; // 0..3
            pairs[2*(pairs_offset+dest_offset+10*grp_offset)+lid] = pairs_prev_offset + indices[prefix_idx+next_idx];
        }
        lid += 5*num_groups*num_chunks;
    }
}

kernel void final5(
                   const global uint* schedule,
                   const uint schedule_offset,
                   const uint n,
                   const global uint* sieve,
                   const global uint* hash_chunks_src,
                   global uint* pairs,
                   const uint pairs_prev_offset,
                   const uint pairs_offset,
                   const uint dest_offset,
                   local uint* sieve_index,
                   local uint* indices,
                   local uint* chunks
                   )
{
    ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/5;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    indices[lid] = sieve[16*sieve_index[lid/5]+(lid%5)];
    barrier(CLK_LOCAL_MEM_FENCE);
    chunks[lid] = hash_chunks_src[2*indices[lid]+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    while (lid < 10*num_groups)
    {
        if (grp_offset+(lid/10) < n)
        {
            const ushort prefix_idx = 5*(lid/10);
            const uchar idx = lid%10; // 0..9
            const uchar my_idx = idx%5; // 0..4
            const uchar next_idx = (my_idx + 1 + (idx/5))%5; // 0..4
            const uchar equal = chunks[prefix_idx+my_idx] == chunks[prefix_idx+next_idx];
            pairs[2*(pairs_offset+dest_offset+10*grp_offset+lid)] = equal ? (pairs_prev_offset + indices[prefix_idx+my_idx]) : 0;
            pairs[2*(pairs_offset+dest_offset+10*grp_offset+lid)+1] = equal ? (pairs_prev_offset + indices[prefix_idx+next_idx]) : 0;
        }
        lid += 5*num_groups;
    }
}

kernel void group_c(
                    const global uint* schedule,
                    const uint schedule_offset,
                    const uint n,
                    const global uint* sieve,
                    const global uint* hash_chunks_src,
                    global uint* hash_chunks_dst,
                    global uint* pairs,
                    const uint pairs_prev_offset,
                    const uint pairs_offset,
                    const uint dest_offset,
                    const uchar num_chunks,
                    const uchar count,
                    local uint* sieve_index,
                    local uint* indices,
                    local uint* chunks,
                    global uint* vchunks
                    )
{
    ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/num_chunks/count;
    const uchar combs = count*(count-1)/2;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups && grp_offset + lid < n)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < count*num_groups && grp_offset + (lid/count) < n)
    {
        indices[lid] = sieve[16*sieve_index[lid/count]+(lid%count)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (grp_offset+(lid/(count*num_chunks)) < n)
    {
        chunks[lid] = hash_chunks_src[(num_chunks+1)*indices[lid/num_chunks]+(lid%num_chunks)+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    while (lid < combs*num_groups*num_chunks)
    {
        if (grp_offset+(lid/(combs*num_chunks)) < n)
        {
            const ushort prefix_idx = count*num_chunks*(lid/(combs*num_chunks));
            const uchar idx = (lid%(combs*num_chunks))/num_chunks; // 0..9
            const uchar my_idx = idx%count; // 0..4
            const uchar next_idx = (my_idx + 1 + (idx/count))%count; // 0..4
            const uint v = chunks[prefix_idx+my_idx*num_chunks+(lid%num_chunks)]^chunks[prefix_idx+next_idx*num_chunks+(lid%num_chunks)];
            if (lid%num_chunks==0)
            {
                vchunks[dest_offset+combs*grp_offset+(lid/num_chunks)] = v;
            }
            else
            {
                hash_chunks_dst[num_chunks*(dest_offset+combs*grp_offset)+lid] = v;
            }
        }
        if ((lid < 2*combs*num_groups) && (grp_offset+((lid/2)/combs) < n))
        {
            const ushort prefix_idx = count*((lid/2)/combs);
            const uchar idx = (lid/2)%combs; // 0..9
            const uchar next_idx = ((idx%count)+(1+(idx/count))*(lid%2))%count; // 0..3
            pairs[2*(pairs_offset+dest_offset+combs*grp_offset)+lid] = pairs_prev_offset + indices[prefix_idx+next_idx];
        }
        lid += count*num_groups*num_chunks;
    }
}

kernel void final_c(
                    const global uint* schedule,
                    const uint schedule_offset,
                    const uint n,
                    const global uint* sieve,
                    const global uint* hash_chunks_src,
                    global uint* pairs,
                    const uint pairs_prev_offset,
                    const uint pairs_offset,
                    const uint dest_offset,
                    const uchar count,
                    local uint* sieve_index,
                    local uint* indices,
                    local uint* chunks
                    )
{
    ushort lid = get_local_id(0);
    const ushort num_groups = get_local_size(0)/count;
    const uchar combs = count*(count-1)/2;
    const uint grp_offset = get_group_id(0)*num_groups;
    if (lid < num_groups && grp_offset + lid < n)
    {
        sieve_index[lid] = schedule[schedule_offset + grp_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (grp_offset + (lid/count) < n)
    {
        indices[lid] = sieve[16*sieve_index[lid/count]+(lid%count)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (grp_offset + (lid/count) < n)
    {
        chunks[lid] = hash_chunks_src[2*indices[lid]+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    while (lid < combs*num_groups)
    {
        if (grp_offset+(lid/combs) < n)
        {
            const ushort prefix_idx = count*(lid/combs);
            const uchar idx = lid%combs; // 0..9
            const uchar my_idx = idx%count; // 0..4
            const uchar next_idx = (my_idx + 1 + (idx/count))%count; // 0..4
            const uchar equal = chunks[prefix_idx+my_idx] == chunks[prefix_idx+next_idx];
            pairs[2*(pairs_offset+dest_offset+combs*grp_offset+lid)] = equal ? (pairs_prev_offset + indices[prefix_idx+my_idx]) : 0;
            pairs[2*(pairs_offset+dest_offset+combs*grp_offset+lid)+1] = equal ? (pairs_prev_offset + indices[prefix_idx+next_idx]) : 0;
        }
        lid += count*num_groups;
    }
}
)";

#endif /* kernels_h */
