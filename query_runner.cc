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
#include <matplot/matplot.h>

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
  float x, y, z, ux, uy, uz, ke;
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
  Decode(file_, &particle->ke);
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
  std::vector<std::string> timestep_name_;
  std::vector<double> timestep_;
  std::vector<double> ke_;
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> z_;
  const std::string particle_dir_;
  const std::string sp_name_;
  int nproc_;
};

QueryProcessor::QueryProcessor(const std::string& dir, const std::string& sp,
                               int nproc)
    : particle_dir_(dir), sp_name_(sp), nproc_(nproc) {}

int QueryProcessor::GetTrajectory(uint64_t target, int first_step,
                                  int last_step, int interval) {
  memset(&stat_, 0, sizeof(QueryStat));
  timestep_name_.resize(0);
  timestep_.resize(0);
  x_.resize(0);
  y_.resize(0);
  z_.resize(0);
  std::string filename = particle_dir_;
  const size_t prefix_len = filename.length();
  uint64_t prev = stat_.particles_fetched;
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
        sprintf(tmp, "step %d", step);
        timestep_name_.push_back(tmp);
        timestep_.push_back(step);
        x_.push_back(part.x);
        y_.push_back(part.y);
        z_.push_back(part.z);
        ke_.push_back(part.ke);
        printf("step=%d, id=%d, x=%f, y=%f, z=%f, ke=%f, particles=%d\n", step,
               int(part.id), part.x, part.y, part.z, part.ke,
               int(stat_.particles_fetched - prev));
        prev = stat_.particles_fetched;
      }
    }
  }
  auto f = matplot::gcf(true);
  f->size(1000, 500);
  matplot::tiledlayout(1, 2);
  auto ax1 = matplot::nexttile();
  matplot::plot(ax1, timestep_, ke_, "-o")->line_width(1.5);
  matplot::xlabel(ax1, "Timestep");
  matplot::ylabel(ax1, "Energy");
  auto ax2 = matplot::nexttile();
  matplot::plot3(ax2, x_, y_, z_, "-o")->line_width(1.5);
  matplot::xlabel(ax2, "x");
  matplot::ylabel(ax2, "y");
  matplot::zlabel(ax2, "z");
  matplot::title(ax2, "Trajectory");
  ax2->box_full(true);
  matplot::show();
  matplot::save("figure.svg");
  return timestep_.size();
}

}  // namespace vpic

static void usage(char* argv0, const char* msg) {
  if (msg) fprintf(stderr, "%s: %s\n\n", argv0, msg);
  fprintf(stderr, "===============\n");
  fprintf(stderr, "Usage: %s [options] particle_dir\n\n", argv0);
  fprintf(stderr, "-f\tint\t\t:  starting timestep\n");
  fprintf(stderr, "-l\tint\t\t:  last timestep\n");
  fprintf(stderr, "-s\tint\t\t:  timesteps to skip between two searches\n");
  fprintf(stderr, "-e\telectron\t\t:  search electron particles\n");
  fprintf(stderr, "-i\tion\t\t:  search ion particles\n");
  fprintf(stderr, "-t\tid\t\t:  ID of the particle to search\n");
  fprintf(stderr, "===============\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
  char* const argv0 = argv[0];
  int first_timestep = 0, last_timestep = 0, interval = 1;
  int electron = 1;
  int ion = 0;
  int64_t target = 0;
  int c;

  setlinebuf(stdout);
  while ((c = getopt(argc, argv, "f:l:s:t:eih")) != -1) {
    switch (c) {
      case 'f':
        first_timestep = atoi(optarg);
        if (first_timestep < 0) usage(argv0, "invalid initial timestep ID");
        break;
      case 'l':
        last_timestep = atoi(optarg);
        if (last_timestep < 0) usage(argv0, "invalid last timestep ID");
        break;
      case 's':
        interval = atoi(optarg);
        if (interval < 1) usage(argv0, "invalid timestep interval");
        break;
      case 'e':
        electron = 1;
        ion = 0;
        break;
      case 'i':
        electron = 0;
        ion = 1;
        break;
      case 't':
        target = atoll(optarg);
        if (target < 0) usage(argv0, "invalid target ID");
        break;
      case 'h':
      default:
        usage(argv0, NULL);
        break;
    }
  }

  argc -= optind;
  argv += optind;

  if (argc < 1) {
    usage(argv0, "must specify particle dir path");
  }
  printf("Search particle %d from %s within [%d,%d,%d]\n", int(target), argv[0],
         first_timestep, last_timestep, interval);
  vpic::QueryProcessor proc(argv[0], electron ? "eparticle" : "iparticle",
                            1 /* TODO */);
  int r = proc.GetTrajectory(target, first_timestep, last_timestep, interval);
  printf("Found %d steps\n", r);
}
