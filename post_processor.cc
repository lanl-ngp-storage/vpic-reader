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

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace vpic {
// Information on a given grid of a VPIC simulation
struct Grid {
  Grid() {}  // Intentionally not initialized for performance
  int step, nx, ny, nz, nxp2, nyp2, nzp2;
  float dt, dx, dy, dz, x0, y0, z0, cvac, eps0, damp;
  int rank, nproc;
  int sp_id;
  float q_m;
};

// Header information written at the beginning of each particle dump
struct Header {
  Header() {}  // Intentionally not initialized for performance
  unsigned char char_bits, shortint_len, int_len, float_len, double_len;
  unsigned short int magic1;
  unsigned int magic2;
  float float_const;
  double double_const;
  int version, dump_type;
  Grid grid;
};

// Information describing a data array
struct ArrayHeader {
  ArrayHeader() {}  // Intentionally not initialized for performance
  int data_size;
  int ndim;
  int dim[3];
};

// Information regarding one specific particle
struct Particle {
  Particle() {}  // Intentionally not initialized for performance
  float dx, dy, dz, x, y, z;
  int i;
  float ux, uy, uz;
  float w;
};

class Reader {
 public:
  explicit Reader(const std::string& filename);
  ~Reader();

  void Open();
  void ReadHeader(Header* header);
  void ReadArrayHeader(ArrayHeader* header);
  void Seek(size_t offset);
  size_t current_pos() const { return ftell(file_); }
  void NextParticle(Particle* particle, const Grid* grid);
  void NextParticleId(uint64_t* id);

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

void Reader::Open() {
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
}

void Reader::ReadHeader(Header* const header) {
  if (!file_) {
    abort();
  }
  Decode(file_, &header->char_bits);
  if (header->char_bits != 8) {
    fprintf(stderr, "Header has %hhu bits per byte, which is scary\n",
            header->char_bits);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->shortint_len);
  if (header->shortint_len != 2) {
    fprintf(stderr, "sizeof(short int) is %hhu (not 2), which is scary\n",
            header->shortint_len);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->int_len);
  if (header->int_len != 4) {
    fprintf(stderr, "sizeof(int) is %hhu (not 4), which is scary\n",
            header->int_len);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->float_len);
  if (header->float_len != 4) {
    fprintf(stderr, "sizeof(float) is %hhu (not 4), which is scary\n",
            header->int_len);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->double_len);
  if (header->double_len != 8) {
    fprintf(stderr, "sizeof(double) is %hhu (not 8), which is scary\n",
            header->int_len);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->magic1);
  if (header->magic1 != 0xcafe) {
    fprintf(stderr, "magic1 is 0x%x (not 0xcafe), so something is wrong\n",
            header->magic1);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->magic2);
  if (header->magic2 != 0xdeadbeef) {
    fprintf(stderr, "magic2 is 0x%x (not 0xdeadbeef), so something is wrong\n",
            header->magic2);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->float_const);
  if (header->float_const != 1.) {
    fprintf(stderr, "float_const is %f (not 1.0), so something is wrong\n",
            header->float_const);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->double_const);
  if (header->double_const != 1.) {
    fprintf(stderr, "double_const is %f (not 1.0), so something is wrong\n",
            header->double_const);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->version);
  if (header->version != 0) {
    fprintf(stderr, "version is %d (not 0), so something is wrong\n",
            header->version);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->dump_type);
  if (header->dump_type < 0 || header->dump_type > 5) {
    fprintf(stderr, "dump_type is %d (not within 0-5), so something is wrong\n",
            header->version);
    exit(EXIT_FAILURE);
  }

  Grid* const gd = &header->grid;
  Decode(file_, &gd->step);
  Decode(file_, &gd->nx);
  Decode(file_, &gd->ny);
  Decode(file_, &gd->nz);
  Decode(file_, &gd->dt);
  Decode(file_, &gd->dx);
  Decode(file_, &gd->dy);
  Decode(file_, &gd->dz);
  Decode(file_, &gd->x0);
  Decode(file_, &gd->y0);
  Decode(file_, &gd->z0);
  Decode(file_, &gd->cvac);
  Decode(file_, &gd->eps0);
  Decode(file_, &gd->damp);
  Decode(file_, &gd->rank);
  Decode(file_, &gd->nproc);
  Decode(file_, &gd->sp_id);
  Decode(file_, &gd->q_m);
  // Storing the following information as cache for followup per-particle
  // location calculation
  gd->nxp2 = gd->nx + 2;
  gd->nyp2 = gd->ny + 2;
  gd->nzp2 = gd->nz + 2;
}

void Reader::ReadArrayHeader(ArrayHeader* const header) {
  if (!file_) {
    abort();
  }
  header->dim[0] = 1;
  header->dim[1] = 1;
  header->dim[2] = 1;
  Decode(file_, &header->data_size);
  if (header->data_size <= 0) {
    fprintf(stderr, "array data size is %d, so something is wrong\n",
            header->data_size);
    exit(EXIT_FAILURE);
  }
  Decode(file_, &header->ndim);
  if (header->ndim < 1 || header->ndim > 3) {
    fprintf(stderr, "ndim is %d (not within 1-3), so something is wrong\n",
            header->ndim);
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < header->ndim; i++) {
    Decode(file_, &header->dim[i]);
    if (header->dim[i] < 1) {
      fprintf(stderr, "dim[%d] is %d, so something is wrong\n", i,
              header->dim[i]);
      exit(EXIT_FAILURE);
    }
  }
}

void Reader::Seek(size_t offset) {
  if (!file_) {
    abort();
  }
  fseek(file_, offset % file_size_, SEEK_SET);
}

void Reader::NextParticle(Particle* particle, const Grid* grid) {
  assert(file_);
  Decode(file_, &particle->dx);
  Decode(file_, &particle->dy);
  Decode(file_, &particle->dz);
  Decode(file_, &particle->i);
  Decode(file_, &particle->ux);
  Decode(file_, &particle->uy);
  Decode(file_, &particle->uz);
  Decode(file_, &particle->w);
  // Turn index i into separate ix, iy, iz indices
  int iz = particle->i / (grid->nxp2 * grid->nyp2);
  int iy = (particle->i - iz * grid->nxp2 * grid->nyp2) / grid->nxp2;
  int ix = particle->i - grid->nxp2 * (iy + grid->nyp2 * iz);
  // Transform local coordinates to global ones
  particle->x = grid->x0 + ((ix - 1) + (particle->dx + 1) * 0.5) * grid->dx;
  particle->y = grid->y0 + ((iy - 1) + (particle->dy + 1) * 0.5) * grid->dy;
  particle->z = grid->z0 + ((iz - 1) + (particle->dz + 1) * 0.5) * grid->dz;
}

void Reader::NextParticleId(uint64_t* id) {
  assert(file_);
  Decode(file_, id);
}

Reader::~Reader() {
  if (file_) {
    fclose(file_);
  }
}

class PostProcessor {
 public:
  PostProcessor(const std::string& in, const std::string& out);
  ~PostProcessor();

  void Open();
  void Prepare();
  void RewriteParticles();
  int nparticles() const { return nparticles_; }

 private:
  Header particle_header_;
  const std::string particle_file_;
  Reader* particle_;
  Reader* particle_id_;
  const std::string outputname_;
  FILE* output_;
  int nparticles_;
};

template <typename T>
inline void Encode(FILE* file, const T& result) {
  int r = fwrite(&result, sizeof(T), 1, file);
  if (r != 1) {
    fprintf(stderr, "Error writing data to file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
}

PostProcessor::PostProcessor(const std::string& in, const std::string& out)
    : particle_file_(in),
      particle_(new Reader(in)),
      particle_id_(new Reader(in)),
      outputname_(out),
      output_(NULL),
      nparticles_(0) {}

void PostProcessor::Open() {
  particle_->Open();
  particle_id_->Open();
  if (output_) {
    return;
  }
  output_ = fopen(outputname_.c_str(), "w");  // Discard existing data
  if (!output_) {
    fprintf(stderr, "Fail to open file for writing %s: %s\n",
            outputname_.c_str(), strerror(errno));
    exit(EXIT_FAILURE);
  }
}

void PostProcessor::Prepare() {
  ArrayHeader ah, ah1;
  particle_->ReadHeader(&particle_header_);
  particle_->ReadArrayHeader(&ah);
  nparticles_ = ah.dim[0] * ah.dim[1] * ah.dim[2];
  if (nparticles_ < 0) {
    abort();
  }
  const size_t particle_start = particle_->current_pos();
  particle_id_->Seek(particle_start + ah.data_size * nparticles_);
  particle_id_->ReadArrayHeader(&ah1);
  if (nparticles_ != ah1.dim[0] * ah1.dim[1] * ah1.dim[2]) {
    abort();
  }
}

void PostProcessor::RewriteParticles() {
  Particle particle;
  uint64_t id;
  uint64_t padding = 0;
  for (int i = 0; i < nparticles_; i++) {
    particle_->NextParticle(&particle, &particle_header_.grid);
    particle_id_->NextParticleId(&id);
    Encode(output_,
           id);  // We add padding below to make each particle id 16 bytes
    Encode(output_, padding);
    Encode(output_, particle.x);
    Encode(output_, particle.y);
    Encode(output_, particle.z);
    Encode(output_, particle.i);
    Encode(output_, particle.ux);
    Encode(output_, particle.uy);
    Encode(output_, particle.uz);
    // Calculate the kinetic energy value of each particle
    double gam2 = 1.0 + particle.ux * particle.ux + particle.uy * particle.uy +
                  particle.uz * particle.uz;
    float ke = static_cast<float>(sqrt(gam2) - 1.0);
    Encode(output_, ke);
  }
}

PostProcessor::~PostProcessor() {
  delete particle_;
  delete particle_id_;
  if (output_) {
    fflush(output_);
    fclose(output_);
  }
}

}  // namespace vpic

void process_file(const char* in, const char* out) {
  std::string tmp;
  if (!out) {
    tmp = in;
    tmp += ".bin";
    out = tmp.c_str();
  }
  vpic::PostProcessor pp(in, out);
  pp.Open();
  pp.Prepare();
  pp.RewriteParticles();
  printf("<< %s\n>> %s (%d particles processed)\n", in, out, pp.nparticles());
}

void process_dir(const char* inputdir, const char* outputdir) {
  DIR* const dir = opendir(inputdir);
  if (!dir) {
    fprintf(stderr, "Fail to open dir %s: %s\n", inputdir, strerror(errno));
    exit(EXIT_FAILURE);
  }
  std::string tmpsrc = inputdir, tmpdst = outputdir;
  size_t tmpsrc_prefix = tmpsrc.length(), tmpdst_prefix = tmpdst.length();
  struct dirent* entry = readdir(dir);
  while (entry) {
    if (entry->d_type == DT_REG && strcmp(entry->d_name, ".") != 0 &&
        strcmp(entry->d_name, "..") != 0) {
      tmpsrc.resize(tmpsrc_prefix);
      tmpsrc += "/";
      tmpsrc += entry->d_name;
      tmpdst.resize(tmpdst_prefix);
      tmpdst += "/";
      tmpdst += entry->d_name;
      tmpdst += ".bin";
      process_file(tmpsrc.c_str(), tmpdst.c_str());
    }
    entry = readdir(dir);
  }
  closedir(dir);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    abort();
  }
  struct stat filestat;
  int r = ::stat(argv[1], &filestat);
  if (r != 0) {
    fprintf(stderr, "Fail to stat file %s: %s\n", argv[1], strerror(errno));
    exit(EXIT_FAILURE);
  }
  if (S_ISREG(filestat.st_mode)) {
    process_file(argv[1], NULL);
  } else if (S_ISDIR(filestat.st_mode)) {
    if (argc < 3) {
      abort();
    }
    process_dir(argv[1], argv[2]);
  } else {
    fprintf(stderr, "Unexpected file type: %s\n", argv[1]);
    exit(EXIT_FAILURE);
  }
  return 0;
}
