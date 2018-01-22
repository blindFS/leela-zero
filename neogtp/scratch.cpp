#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <sstream>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

int main()
{
    std::stringstream res;
    res.str(exec("./gpuload"));
    std::string segment;
    std::vector<std::string> strs;
    while(std::getline(res, segment, '\n')){
        strs.push_back(segment);
    }
    std::cout << std::stoi(strs.at(0));
}
