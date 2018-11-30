#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <ctime>

#include <pthread.h>



pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t start_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t condition =  PTHREAD_COND_INITIALIZER;

bool value_updated =  true;
bool start = false;
bool finish = false;

int sleep_time;
int threads_num;


std::vector<std::string> split(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter))
  {
    tokens.push_back(token);
  }
  return tokens;
  }

std::vector<int> get_values_list(){
  std::string str = "";
    std::getline(std::cin, str);
    std::vector<std::string> nums_str = split(str,' ');
    std::vector<int> nums;
    for (std::string s : nums_str){
      nums.push_back(atoi(s.c_str()));
     }
     return nums;
}
 
class Value {
public:
    Value() : _value(0) { }
 
    void update(int value) {
        _value = value;
    }
 
    int get() const {
        return _value;
    }
 
private:
    int _value;
};
 
void* producer_routine(void* arg) {
  std::vector<int> numlist =  get_values_list();
  Value* v =  static_cast<Value*>(arg);

  while (!(numlist.empty())){
      pthread_mutex_lock(&mutex);
        pthread_cond_wait(&condition, &mutex); 
        (*v).update(numlist.back());
        value_updated = true;
      pthread_mutex_unlock(&mutex);
      numlist.pop_back();    
  }
  finish = true;
}
 
void* consumer_routine(void* arg) {
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);

  Value* v =  static_cast<Value*>(arg);
  int sum = 0;
  srand( time( 0 ) );
  while(!finish){
    pthread_mutex_lock(&mutex);
      if(value_updated){
         sum+=(*v).get(); 
         value_updated = false;
         pthread_cond_signal( &condition_var );
      }
    pthread_mutex_unlock(&mutex);
    usleep((rand() % sleep_time +1 )*1000);
  }
  void* ptr = &sum;
  return ptr;
}
 
void* consumer_interruptor_routine(void* arg) {
  ////
}
 
int run_threads() {
  Value value;

 ///////
  return 0;
}
 
int main(int argc, char *argv[]) {
  if(argc < 3){
    std::cout << "ERROR: must specify two arguments (N - number of consumers, T - max sleeping time)" << std::endl; 
    return 1;
  } 
  threads_num = atoi(argv[1]); 
  sleep_time = atoi(argv[2]);
  std::cout << run_threads() << std::endl;
  return 0;
}
