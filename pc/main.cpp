#include "public_header.h"

int main(int argc, char *argv[]) {
  (void) argc;
  (void) argv;
  std::cout << "iosream is included in public_header.h" << std::endl;

  std::cout << temp::add(1, 2) << std::endl;

  return 0;
}
