//
//  zcash_interface.cpp
//  FXTrue
//
//  Created by Alexey Akhunov on 29/10/2016.
//  Copyright © 2016 Alexey Akhunov. All rights reserved.
//

#include "zcash_interface.hpp"
#include <iostream>
#include <string>
#include <map>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include "rpcprotocol.h"
#include "utilstrencodings.h"
#include "json.hpp"
#include "json_spirit_value.h"
#include "uint256.h"
#include "hash.hpp"
#include "serialize.hpp"

json::Object request(boost::iostreams::stream< SSLIOStreamDevice<boost::asio::ip::tcp> >& stream, std::string strRequest)
{
    std::string strUserPass64 = EncodeBase64("username:password");
    std::map<std::string, std::string> mapRequestHeaders;
    mapRequestHeaders["Authorization"] = std::string("Basic ") + strUserPass64;
    std::string strPost = HTTPPost(strRequest, mapRequestHeaders);
    std::cout << strPost;
    stream << strPost << std::flush;
    // Receive HTTP reply status
    int nProto = 0;
    int nStatus = ReadHTTPStatus(stream, nProto);
    std::map<std::string, std::string> mapHeaders;
    std::string strReply;
    ReadHTTPMessage(stream, mapHeaders, strReply, nProto, std::numeric_limits<size_t>::max());
    std::cout << "STATUS RECEIVED: " << nStatus << "\n";
    std::cout << "REPLY:\n" << strReply << "\n";
    if (nStatus == HTTP_UNAUTHORIZED)
    {
        std::cerr << "incorrect rpcuser or rpcpassword (authorization failed)\n";
        exit(1);
    }
    else if (nStatus >= 400 && nStatus != HTTP_BAD_REQUEST && nStatus != HTTP_NOT_FOUND && nStatus != HTTP_INTERNAL_SERVER_ERROR)
    {
        std::cerr << "server returned HTTP error " << nStatus << "\n";
        exit(1);
    }
    else if (strReply.empty())
    {
        std::cerr << "no response from server\n";
        exit(1);
    }
    
    // Parse reply
    json::Value valReply;
    if (!json::read_string(strReply, valReply))
    {
        std::cerr << "couldn't parse reply from server\n" << strReply << "\n";
        exit(1);
    }
    const json::Object& reply = valReply.get_obj();
    if (reply.empty())
    {
        std::cerr << "expected reply to have result, error and id properties\n";
        exit(1);
    }
    return reply;
}

uint256 buildMerkleTree(const std::vector<std::string> txHashes)
{
    /* WARNING! If you're reading this because you're learning about crypto
     and/or designing a new system that will use merkle trees, keep in mind
     that the following merkle tree algorithm has a serious flaw related to
     duplicate txids, resulting in a vulnerability (CVE-2012-2459).
     The reason is that if the number of hashes in the list at a given time
     is odd, the last one is duplicated before computing the next level (which
     is unusual in Merkle trees). This results in certain sequences of
     transactions leading to the same merkle root. For example, these two
     trees:
     A               A
     /  \            /   \
     B     C         B       C
     / \    |        / \     / \
     D   E   F       D   E   F   F
     / \ / \ / \     / \ / \ / \ / \
     1 2 3 4 5 6     1 2 3 4 5 6 5 6
     for transaction lists [1,2,3,4,5,6] and [1,2,3,4,5,6,5,6] (where 5 and
     6 are repeated) result in the same root hash A (because the hash of both
     of (F) and (F,F) is C).
     The vulnerability results from being able to send a block with such a
     transaction list, with the same merkle root, and the same block hash as
     the original without duplication, resulting in failed validation. If the
     receiving node proceeds to mark that block as permanently invalid
     however, it will fail to accept further unmodified (and thus potentially
     valid) versions of the same block. We defend against this by detecting
     the case where we would hash two identical hashes at the end of the list
     together, and treating that identically to the block having an invalid
     merkle root. Assuming no double-SHA256 collisions, this will detect all
     known ways of changing the transactions without affecting the merkle
     root.
     */
    std::vector<uint256> vMerkleTree;
    vMerkleTree.reserve(txHashes.size() * 2 + 16); // Safe upper bound for the number of total nodes.
    for (const auto& txHash: txHashes)
        vMerkleTree.push_back(uint256S(txHash));
    size_t j = 0;
    for (size_t nSize = txHashes.size(); nSize > 1; nSize = (nSize + 1) / 2)
    {
        for (size_t i = 0; i < nSize; i += 2)
        {
            size_t i2 = std::min(i+1, nSize-1);
            vMerkleTree.push_back(Hash(BEGIN(vMerkleTree[j+i]),  END(vMerkleTree[j+i]),
                                       BEGIN(vMerkleTree[j+i2]), END(vMerkleTree[j+i2])));
        }
        j += nSize;
    }
    return (vMerkleTree.empty() ? uint256() : vMerkleTree.back());
}

block_template getblocktemplate() {
    std::cout << "GETBLOCKTEMPLATE\n";
    bool fUseSSL = false;
    boost::asio::io_service io_service;
    boost::asio::ssl::context context(io_service, boost::asio::ssl::context::sslv23);
    context.set_options(boost::asio::ssl::context::no_sslv2 | boost::asio::ssl::context::no_sslv3);
    boost::asio::ssl::stream<boost::asio::ip::tcp::socket> sslStream(io_service, context);
    SSLIOStreamDevice<boost::asio::ip::tcp> d(sslStream, fUseSSL);
    boost::iostreams::stream< SSLIOStreamDevice<boost::asio::ip::tcp> > stream(d);
    const bool fConnected = d.connect("192.168.1.106", "18232");
    std::cout << "CONNECTED TO RPC: " << fConnected << "\n";
    std::string getblocktemplate_req = R"(
        {"id": 0, "method": "getblocktemplate", "params": []}
    )";
    json::Object getblocktemplate_res = request(stream, getblocktemplate_req);
    block_template bt;
    std::string coinbase_hash;
    std::string coinbase_data;
    std::vector<std::string> txHashes;
    std::vector<std::string> txData;
    for(const auto& item: getblocktemplate_res)
    {
        if (json::Config::get_name(item) == "result")
        {
            const json::Object result = json::Config::get_value(item).get_obj();
            for(const auto& item1: result)
            {
                if (json::Config::get_name(item1) == "coinbasetxn")
                {
                    const json::Object coinbasetxn = json::Config::get_value(item1).get_obj();
                    for(const auto& item2: coinbasetxn)
                    {
                        if (json::Config::get_name(item2) == "data")
                        {
                            txData.insert(txData.begin(), json::Config::get_value(item2).get_str());
                            coinbase_data = json::Config::get_value(item2).get_str();
                        }
                        else if (json::Config::get_name(item2) == "hash")
                        {
                            txHashes.insert(txHashes.begin(), json::Config::get_value(item2).get_str());
                            coinbase_hash = json::Config::get_value(item2).get_str();
                        }
                    }
                }
                else if (json::Config::get_name(item1) == "transactions")
                {
                    json::Array txs = json::Config::get_value(item1).get_array();
                    for(const auto& item2: txs)
                    {
                        json::Object tx = item2.get_obj();
                        for(const auto& item3: tx)
                        {
                            if (json::Config::get_name(item3) == "data")
                            {
                                txData.push_back(json::Config::get_value(item3).get_str());
                            }
                            else if (json::Config::get_name(item3) == "hash")
                            {
                                txHashes.push_back(json::Config::get_value(item3).get_str());
                            }
                        }
                    }
                }
                else if (json::Config::get_name(item1) == "version")
                {
                    bt.version = json::Config::get_value(item1).get_int();
                }
                else if (json::Config::get_name(item1) == "curtime")
                {
                    bt.curtime = json::Config::get_value(item1).get_int();
                }
                else if (json::Config::get_name(item1) == "bits")
                {
                    bt.bits = std::stoi(json::Config::get_value(item1).get_str(), nullptr, 16);
                }
                else if (json::Config::get_name(item1) == "target")
                {
                    bt.hashTarget = uint256S(json::Config::get_value(item1).get_str());
                }
                else if (json::Config::get_name(item1) == "previousblockhash")
                {
                    bt.prevBlockHash = uint256S(json::Config::get_value(item1).get_str());
                }
            }
        }
    }
    bt.merkleRoot = buildMerkleTree(txHashes);
    std::cout << "MY MERKLE ROOT: " << bt.merkleRoot.ToString() << "\n";
    std::cout << "VERSION: " << bt.version << "\n";
    std::cout << "TIME: " << bt.curtime << "\n";
    std::cout << "BITS: " << bt.bits << "\n";
    std::cout << "PREV_BLOCK_HASH: " << bt.prevBlockHash.ToString() << "\n";
    std::stringstream block_header_stream;
    Serialize(block_header_stream, bt.version, 0, 0);
    bt.prevBlockHash.Serialize(block_header_stream, 0, 0);
    bt.merkleRoot.Serialize(block_header_stream, 0, 0);
    uint256().Serialize(block_header_stream, 0, 0);
    Serialize(block_header_stream, bt.curtime, 0, 0);
    Serialize(block_header_stream, bt.bits, 0, 0);
    std::string header_template = block_header_stream.str();
    std::copy(header_template.begin(), header_template.end(), bt.header_template);
    std::cout << "HASH TARGET: " << bt.hashTarget.ToString() << "\n";
    return bt;
}

uint256 block_template::blockHash(const std::vector<unsigned char>& sol, const uint256& nonce)
{
    std::stringstream block_header_stream;
    Serialize(block_header_stream, version, 0, 0);
    prevBlockHash.Serialize(block_header_stream, 0, 0);
    merkleRoot.Serialize(block_header_stream, 0, 0);
    uint256().Serialize(block_header_stream, 0, 0);
    Serialize(block_header_stream, curtime, 0, 0);
    Serialize(block_header_stream, bits, 0, 0);
    nonce.Serialize(block_header_stream, 0, 0);
    Serialize(block_header_stream, sol, 0, 0);
    std::string block_header = block_header_stream.str();
    return Hash(block_header.cbegin(), block_header.cend());
}
