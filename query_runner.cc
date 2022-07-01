/*
 * Copyright (c) 2021 Triad National Security, LLC, as operator of Los Alamos
 * National Laboratory with the U.S. Department of Energy/National Nuclear
 * Security Administration. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of TRIAD, Los Alamos National Laboratory, LANL, the
 *    U.S. Government, nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace vpic {
// Query status information
struct QueryStat {
  QueryStat() : files_opened(0), particles_fetched(0), bytes_read(0) {}
  uint64_t files_opened;
  uint64_t particles_fetched;
  uint64_t bytes_read;
};

// Information about a specific particle
struct Particle {
  Particle() {}  // Intentionally not initialized for performance
  uint64_t id;
  float x, y, z, ux, uy, uz, w;
  int i;
};

class Reader {
 public:
  explicit Reader(const std::string& filename);
  ~Reader();
  void Open(QueryStat* stat);
  void NextParticle(Particle* particle, QueryStat* stat);
  bool has_next() const { return (ftell(file_) + 48) < file_size_; }

 private:
  // No copying allowed
  Reader(const Reader&);
  void operator=(const Reader& reader);
  const std::string filename_;
  size_t file_size_;
  FILE* file_;
};

template <typename T>
inline void Decode(FILE* file, T* result) {
  int r = fread(result, sizeof(T), 1, file);
  if (r != 1) {
    fprintf(stderr, "Error reading data from file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
}

Reader::Reader(const std::string& filename)
    : filename_(filename), file_size_(0), file_(NULL) {}

void Reader::Open(QueryStat* stat) {
  if (file_) {
    return;
  }
  file_ = fopen(filename_.c_str(), "r");
  if (!file_) {
    fprintf(stderr, "Fail to open file %s: %s\n", filename_.c_str(),
            strerror(errno));
    exit(EXIT_FAILURE);
  }
  int r = fseek(file_, 0, SEEK_END);
  if (r != 0) {
    fprintf(stderr, "Error seeking to the end of file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  file_size_ = ftell(file_);
  rewind(file_);
  stat->files_opened++;
}

void Reader::NextParticle(Particle* particle, QueryStat* stat) {
  assert(file_);
  uint64_t ignored_padding;
  Decode(file_, &particle->id);
  Decode(file_, &ignored_padding);
  Decode(file_, &particle->x);
  Decode(file_, &particle->y);
  Decode(file_, &particle->z);
  Decode(file_, &particle->i);
  Decode(file_, &particle->ux);
  Decode(file_, &particle->uy);
  Decode(file_, &particle->uz);
  Decode(file_, &particle->w);
  stat->particles_fetched += 1;
  stat->bytes_read += 48;
}

Reader::~Reader() {
  if (file_) {
    fclose(file_);
  }
}

class Searcher {
 public:
  explicit Searcher(const std::string& filename);
  ~Searcher();
  bool Lookup(uint64_t target, Particle* particle, QueryStat* stat);

 private:
  // No copying allowed
  Searcher(const Searcher&);
  void operator=(const Searcher& searcher);
  Reader* reader_;
};

Searcher::Searcher(const std::string& filename)
    : reader_(new Reader(filename)) {}

bool Searcher::Lookup(uint64_t target, Particle* particle, QueryStat* stat) {
  reader_->Open(stat);
  memset(particle, 0, sizeof(Particle));
  particle->id = ~static_cast<uint64_t>(0);
  while (reader_->has_next()) {
    reader_->NextParticle(particle, stat);
    if (particle->id == target) {
      break;
    }
  }
  return target == particle->id;
}

Searcher::~Searcher() {
  delete reader_;  // This will close the particle file
}

class QueryProcessor {
 public:
  QueryProcessor(const std::string& particle_dir, const std::string& sp_name,
                 int nproc);
  int GetTrajectory(uint64_t target, int first_step, int last_step,
                    int interval);

 private:
  QueryStat stat_;
  std::vector<std::pair<int, Particle> > trajectory_;
  const std::string particle_dir_;
  const std::string sp_name_;
  int nproc_;
};

QueryProcessor::QueryProcessor(const std::string& dir, const std::string& sp,
                               int nproc)
    : particle_dir_(dir), sp_name_(sp), nproc_(nproc) {}

int QueryProcessor::GetTrajectory(uint64_t target, int first_step,
                                  int last_step, int interval) {
  std::string filename = particle_dir_;
  const size_t prefix_len = filename.length();
  memset(&stat_, 0, sizeof(QueryStat));
  trajectory_.resize(0);
  for (int step = first_step; step <= last_step; step += interval) {
    for (int i = 0; i < nproc_; i++) {
      char tmp[100];
      sprintf(tmp, "%s.%d.%d.bin", sp_name_.c_str(), step, i);
      filename.resize(prefix_len);
      filename += "/";
      filename += tmp;
      Searcher searcher(filename);
      Particle part;
      if (searcher.Lookup(target, &part, &stat_)) {
        trajectory_.push_back(std::make_pair(step, part));
        printf("step=%d, id=%lld, x=%f, y=%f, z=%f\n", step, part.id, part.x,
               part.y, part.z);
      }
    }
  }
  return trajectory_.size();
}

}  // namespace vpic

int main(int argc, char* argv[]) {
  vpic::QueryProcessor proc(argv[1], "iparticle", 1);
  int r = proc.GetTrajectory(atoll(argv[2]), 0, 480, 24);
  printf("Found %d steps\n", r);
}
