//
//  prefix_sum.hpp
//  FXTrue
//
//  Created by Alexey Akhunov on 28/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#ifndef prefix_sum_h
#define prefix_sum_h

void prefix_sum(cl::CommandQueue& queue,
                cl::Buffer& input,
                const cl_uint input_size,
                const cl_uint max_workgroup_size,
                cl::Buffer& partial_one,
                cl::Buffer& partial_two,
                cl::Kernel& scan_subarrays_kernel,
                std::vector<cl::Event>& subarray_events,
                cl::Kernel& scan_inc_subarrays_kernel,
                std::vector<cl::Event>& subarray_inc_events
                )
{
    // Scan the counts
    const cl_uint scan_batch = max_workgroup_size * 2;
    const cl_uint segments_one = (input_size+scan_batch-1)/scan_batch;
    const cl_uint scan_size = segments_one*max_workgroup_size;
    // Partial sums
    check(__LINE__,scan_subarrays_kernel.setArg(0, cl::Local(sizeof(cl_uint)*(max_workgroup_size+(max_workgroup_size>>4))*2)));
    check(__LINE__,scan_subarrays_kernel.setArg(1, input));
    check(__LINE__,scan_subarrays_kernel.setArg(2, partial_one));
    check(__LINE__,scan_subarrays_kernel.setArg(3, input_size));
    subarray_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(scan_subarrays_kernel, cl::NullRange, cl::NDRange{scan_size}, cl::NDRange{max_workgroup_size}, nullptr, &subarray_events.back()));
    const cl_uint segments_two = (segments_one + scan_batch - 1) / scan_batch;
    const cl_uint scan_size_one = segments_two*max_workgroup_size;
    check(__LINE__,scan_subarrays_kernel.setArg(0, cl::Local(sizeof(cl_uint)*(max_workgroup_size+(max_workgroup_size>>4))*2)));
    check(__LINE__,scan_subarrays_kernel.setArg(1, partial_one));
    check(__LINE__,scan_subarrays_kernel.setArg(2, partial_two));
    check(__LINE__,scan_subarrays_kernel.setArg(3, segments_one));
    subarray_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(scan_subarrays_kernel, cl::NullRange, cl::NDRange{scan_size_one}, cl::NDRange{max_workgroup_size}, nullptr, &subarray_events.back()));
    if (segments_one > 2*max_workgroup_size) {
        check(__LINE__,scan_subarrays_kernel.setArg(0, cl::Local(sizeof(cl_uint)*(max_workgroup_size+(max_workgroup_size>>4))*2)));
        check(__LINE__,scan_subarrays_kernel.setArg(1, partial_two));
        check(__LINE__,scan_subarrays_kernel.setArg(2, partial_two)); // This won't be written
        check(__LINE__,scan_subarrays_kernel.setArg(3, segments_two));
        const cl_uint scan_size_two = ((segments_two+scan_batch-1)/scan_batch)*max_workgroup_size;
        subarray_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(scan_subarrays_kernel, cl::NullRange, cl::NDRange{scan_size_two}, cl::NDRange{max_workgroup_size}, nullptr, &subarray_events.back()));
        check(__LINE__,scan_inc_subarrays_kernel.setArg(0, cl::Local(sizeof(cl_uint)*max_workgroup_size*2)));
        check(__LINE__,scan_inc_subarrays_kernel.setArg(1, partial_two));
        check(__LINE__,scan_inc_subarrays_kernel.setArg(2, partial_two));
        check(__LINE__,scan_inc_subarrays_kernel.setArg(3, segments_two));
        subarray_inc_events.emplace_back();
        check(__LINE__,queue.enqueueNDRangeKernel(scan_inc_subarrays_kernel, cl::NullRange, cl::NDRange{scan_size_two}, cl::NDRange{max_workgroup_size}, nullptr, &subarray_inc_events.back()));
    }
    check(__LINE__,scan_inc_subarrays_kernel.setArg(0, cl::Local(sizeof(cl_uint)*max_workgroup_size*2)));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(1, partial_one));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(2, partial_two));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(3, segments_one));
    subarray_inc_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(scan_inc_subarrays_kernel, cl::NullRange, cl::NDRange{scan_size_one}, cl::NDRange{max_workgroup_size}, nullptr, &subarray_inc_events.back()));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(0, cl::Local(sizeof(cl_uint)*max_workgroup_size*2)));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(1, input));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(2, partial_one));
    check(__LINE__,scan_inc_subarrays_kernel.setArg(3, input_size));
    subarray_inc_events.emplace_back();
    check(__LINE__,queue.enqueueNDRangeKernel(scan_inc_subarrays_kernel, cl::NullRange, cl::NDRange{scan_size}, cl::NDRange{max_workgroup_size}, nullptr, &subarray_inc_events.back()));
}

#endif /* prefix_sum_h */
