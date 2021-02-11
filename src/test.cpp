#include <iostream>
#include <cmath>

int main()
{

  double rand_num;
for(unsigned i=0;i<10;++i){
  rand_num = 0 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (0))));
  std::cout<<rand_num<<"\n";
}

  return 0;




}
