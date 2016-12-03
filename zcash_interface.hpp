//
//  zcash_interface.hpp
//  FXTrue
//
//  Created by Alexey Akhunov on 29/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#ifndef zcash_interface_hpp
#define zcash_interface_hpp

#include "uint256.h"

struct block_template
{
    unsigned char header_template[108]; // Header to work on
    uint256 hashTarget; // Difficulty target;
    int32_t version;
    uint256 prevBlockHash;
    uint256 merkleRoot;
    uint32_t curtime;
    uint32_t bits;
    
    uint256 blockHash(const std::vector<unsigned char>& sol, const uint256& nonce);
};

block_template getblocktemplate();


#endif /* zcash_interface_hpp */
