//
//  main.cpp
//  FXTrue
//
//  Created by Alexey Akhunov on 27/09/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#include <iostream>
#include <chrono>
#include "equihash.h"
#include "uint256.h"
#include <cstdio>
#include "zcash_interface.hpp"
#include "opencl_solve.hpp"

int main(int argc, const char * argv[]) {
    block_template bt = getblocktemplate();
    eh_HashState base_state;
    Eh200_9.InitialiseState(base_state);
    std::vector<std::vector<unsigned char>> mysolutions;
    OpenCLSolver solver{[&mysolutions,&bt,&base_state](const std::vector<unsigned char>& soln, const uint256& nonce) {
        eh_HashState state = base_state;
        crypto_generichash_blake2b_update(&state, bt.header_template, sizeof(bt.header_template));
        crypto_generichash_blake2b_update(&state, nonce.begin(), nonce.size());
        assert(Eh200_9.IsValidSolution(state, soln));
        uint256 blockHash = bt.blockHash(soln, nonce);
        std::cout << blockHash.ToString() << "\n";
        if (blockHash < bt.hashTarget)
        {
            std::cout << "CORRECT BLOCK HASH FOUND: " << blockHash.ToString() << "\n";
            mysolutions.push_back(soln);
            return true;
        }
        return false;
    }, [](EhSolverCancelCheck pos) {
        return false;
    }};
    solver.run(bt.header_template);
    solver.print_timings();
}
