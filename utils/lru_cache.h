#pragma once

#include <list>
#include <map>
#include <mutex>

/** 
 * @brief A thread-safe LRU (Least Recently Used) cache implementation.
 * This cache uses a combination of a doubly linked list and a map.
 * * @note This implementation uses mutex, so it is not suitable high contention scenarios.
 */

template <typename Key, typename Value>
class LRUCache {
public:
    using ListIt = typename std::list<std::pair<Key, Value>>::iterator;

    explicit LRUCache(size_t capacity) : _capacity(capacity) {}

    bool get(const Key& key, Value& value_out) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto itr = _cache.find(key);

        // If the key is not found, return false
        if (itr == _cache.end()) {
            return false;
        }

        // Move the accessed item to the front of the LRU list
        value_out = itr->second->second;
        _lru_list.erase(itr->second);
        _cache.erase(itr);

        _lru_list.push_front({key, value_out});
        _cache[key] = _lru_list.begin();

        return true;
    }

    void put(const Key& key, const Value& value) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto itr = _cache.find(key);
        // If the key already exists, update its value and move it to the front
        if (itr != _cache.end()) {
            _lru_list.erase(itr->second);
            _cache.erase(itr);
        }

        _lru_list.push_front({key, value});
        _cache[key] = _lru_list.begin();

        // If the cache exceeds capacity, remove the least recently used item
        if (_cache.size() > _capacity) {
            auto removed_itr = _cache.find(_lru_list.rbegin()->first);
            _cache.erase(removed_itr);
            _lru_list.pop_back();
        }
    }
private:
    size_t _capacity;
    std::list<std::pair<Key, Value>> _lru_list;
    std::map<Key, ListIt> _cache;
    std::mutex _mutex;
};