#include <iostream>
#include <pthread.h>
#include <sstream>

#include <unistd.h>
#include <cmath>

class Value;

pthread_mutex_t get_mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t udate_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t read_cond = PTHREAD_COND_INITIALIZER;

bool toggle = true;


int time_to_sleep = 0;
int N;
bool end_of_read = false;

pthread_mutex_t sum_mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  sum_cond;
int all_sum = 0;
bool change = true;



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


void* producer_routine(void* arg)
{
    // Wait for consumer to start
    // Read data, loop through each value and update the value, notify consumer, wait for consumer to process
    Value* var = reinterpret_cast<Value*>(arg);
    std::string input_line;
    
    std::getline(std::cin, input_line);
    int input_value = 0;
    std::istringstream is(input_line);
    
    while (is >> input_value) {
        pthread_mutex_lock(&get_mut);
        while (!toggle) {
            pthread_cond_wait(&read_cond, &get_mut);
        }
        var->update(input_value);
        toggle = !toggle;
        pthread_cond_signal(&udate_cond);
        pthread_mutex_unlock(&get_mut);
    }
    
    pthread_mutex_lock(&get_mut);
    while (!toggle) {
        pthread_cond_wait(&read_cond, &get_mut);
    }
    var->update(0);
    toggle = !toggle;
    end_of_read = true;
    pthread_cond_broadcast(&udate_cond);
    pthread_mutex_unlock(&get_mut);
    pthread_exit(NULL);
}

int get_and_set_sum(int change, int my_val) {
    int curent_sum = 0;
    pthread_mutex_lock(&sum_mut);
    
    pthread_cond_wait(&sum_cond, &sum_mut);
    all_sum += change;
    curent_sum = all_sum;
    
    pthread_cond_broadcast(&sum_cond);
    pthread_mutex_unlock(&sum_mut);
    return curent_sum;
}

int get_val(Value* var) {
    int change = 0;
    pthread_mutex_lock(&get_mut);
    while (toggle) {
        pthread_cond_wait(&udate_cond, &get_mut);
    }
    change = var->get();
    pthread_cond_broadcast(&read_cond);
    if (!end_of_read) {
        toggle = !toggle;
    }
    pthread_cond_broadcast(&read_cond);
    pthread_mutex_unlock(&get_mut);
    return change;
}


void* consumer_routine(void* arg) {
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
    Value* var = reinterpret_cast<Value*>(arg);
    int *my_val = new int();
    *my_val = 0;
    int change = 0;
    while (!end_of_read) {
        change = get_val(var);
        my_val += change;
        *my_val = get_and_set_sum(change, *my_val);
        change  = 0;
        
        int sleep = rand() % time_to_sleep;
        usleep(sleep * 1000);
    }
    if (*my_val != all_sum) {
        *my_val = get_and_set_sum(0, *my_val);
    }
    pthread_exit((void *)my_val);
}


void* consumer_interruptor_routine(void* arg) {
    // wait for consumer to start
    // interrupt consumer while producer is running
    pthread_t *consumers = reinterpret_cast<pthread_t*>(arg);
    int new_rand;
    while (!end_of_read) {
        double r = ((double) rand() / (RAND_MAX));
        new_rand = ceil(r * N);
        pthread_cancel(consumers[new_rand]);
    }
    pthread_exit(NULL);
    
}

int run_threads() {
    // start N threads and wait until they're done
    // return aggregated sum of values
    Value*  variable = new Value();
    
    pthread_t consumers[N];
    pthread_t producer;
    pthread_t interruptor;
    
    for (int i = 0; i < N; ++i) {
        pthread_create(&consumers[i], NULL, consumer_routine, variable);
    }
    
    pthread_create(&interruptor, NULL, consumer_interruptor_routine, consumers);
    pthread_create(&producer, NULL, producer_routine, variable);
    
    pthread_join(producer, NULL);
    pthread_join(interruptor, NULL);
    void* res;
    for (int i = 0; i < N; ++i) {
        pthread_join(consumers[i], &res);
    }
    delete(variable);
    return *((int *)res);
}

int main(int argc, char* argv[]) {
    N = atoi(argv[1]);
    time_to_sleep = atoi(argv[2]);
    std::cout << run_threads() << std::endl;
    return 0;
}
