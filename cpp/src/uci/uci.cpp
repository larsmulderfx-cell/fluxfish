#include "liquid_chess/uci.hpp"
#include <iostream>
#include <sstream>
#include <chrono>

namespace LiquidChess {

class UCIHandler {
private:
    Position pos;
    SearchLimits limits;
    ThreadPool threads;
    TranspositionTable tt;
    
public:
    UCIHandler() : threads(std::thread::hardware_concurrency()),
                   tt(256) { // 256MB TT
        // Initialize engine
        init_magics();
        tt.clear();
    }
    
    void loop() {
        std::string line;
        std::string token;
        
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            iss >> token;
            
            if (token == "uci") {
                identify();
            } else if (token == "isready") {
                std::cout << "readyok" << std::endl;
            } else if (token == "ucinewgame") {
                new_game();
            } else if (token == "position") {
                position(iss);
            } else if (token == "go") {
                go(iss);
            } else if (token == "stop") {
                stop();
            } else if (token == "quit") {
                break;
            } else if (token == "setoption") {
                setoption(iss);
            }
        }
    }
    
private:
    void identify() {
        std::cout << "id name LiquidChess 2.0" << std::endl;
        std::cout << "id author AI Research Team" << std::endl;
        std::cout << "option name Threads type spin default " 
                  << std::thread::hardware_concurrency() 
                  << " min 1 max 128" << std::endl;
        std::cout << "option name Hash type spin default 256 min 1 max 65536" << std::endl;
        std::cout << "option name LRT_Enabled type check default true" << std::endl;
        std::cout << "option name Adaptive_Thinking type check default true" << std::endl;
        std::cout << "uciok" << std::endl;
    }
    
    void position(std::istringstream& iss) {
        std::string token;
        iss >> token;
        
        if (token == "startpos") {
            pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            iss >> token; // Consume "moves" if present
        } else if (token == "fen") {
            std::string fen;
            while (iss >> token && token != "moves") {
                fen += token + " ";
            }
            pos.set(fen);
        }
        
        // Apply moves
        while (iss >> token) {
            Move m = move_from_string(pos, token);
            if (m != MOVE_NONE) {
                pos.do_move(m);
            }
        }
    }
    
    void go(std::istringstream& iss) {
        limits = SearchLimits();
        std::string token;
        
        while (iss >> token) {
            if (token == "wtime") iss >> limits.wtime;
            else if (token == "btime") iss >> limits.btime;
            else if (token == "winc") iss >> limits.winc;
            else if (token == "binc") iss >> limits.binc;
            else if (token == "movestogo") iss >> limits.movestogo;
            else if (token == "depth") iss >> limits.depth;
            else if (token == "nodes") iss >> limits.nodes;
            else if (token == "mate") iss >> limits.mate;
            else if (token == "movetime") iss >> limits.movetime;
            else if (token == "infinite") limits.infinite = true;
        }
        
        // Start search in separate thread
        std::thread([this]() {
            search();
        }).detach();
    }
    
    void search() {
        TimeManager tm(limits, pos.side_to_move());
        Move bestMove = threads.search(pos, MAX_PLY, tm, limits.threads);
        
        std::cout << "bestmove " << move_to_string(bestMove) << std::endl;
    }
    
    void stop() {
        // Signal threads to stop
        threads.stop();
    }
    
    void new_game() {
        tt.clear();
        threads.clear();
    }
    
    void setoption(std::istringstream& iss) {
        std::string token, name, value;
        iss >> token; // Consume "name"
        
        // Read option name
        while (iss >> token && token != "value") {
            if (!name.empty()) name += " ";
            name += token;
        }
        
        // Read option value
        while (iss >> token) {
            if (!value.empty()) value += " ";
            value += token;
        }
        
        // Apply option
        if (name == "Threads") {
            int threads = std::stoi(value);
            this->threads.resize(threads);
        } else if (name == "Hash") {
            int mb = std::stoi(value);
            tt.resize(mb);
        }
    }
};

}