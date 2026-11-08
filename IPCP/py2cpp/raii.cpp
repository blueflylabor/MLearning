#include<iostream>
#include<memory>
#include<vector>
#include<numeric>

using namespace std;
class VectorBuffer{
    public:
        size_t size;
        float* d_ptr;
        VectorBuffer(size_t s):size(s){
            d_ptr = new float[size];
        }
        ~VectorBuffer(){
            delete[] d_ptr;
        }
};

class DeviceTensorBuffer{
    private:
        size_t num_elements_;
        std::unique_ptr<float[]> data_;
    public:
        explicit DeviceTensorBuffer(size_t num_elements) : num_elements_(num_elements), data_(std::make_unique<float[]>(num_elements)){
            std::cout << "[RAII Alloc]: " << num_elements_ * sizeof(float) << "mem." << std::endl;
        }
        ~DeviceTensorBuffer(){
            std::cout << "[Scope Leave]: auto run delete[] mem." << std::endl;
        }
        DeviceTensorBuffer(const DeviceTensorBuffer&) = delete;
        DeviceTensorBuffer(operator=(const DeviceTensorBuffer&)) = delete;
        DeviceTensorBuffer(DeviceTensorBuffer&&) noexcept = default;
        DeviceTensorBuffer& operator=(DeviceTensorBuffer&&) noexcept = default;
        float* data(){
            return data_.get();
        }
        const float* data() const{
            return data_.get();
        }
        size_t size() const{
            return num_elements_;
        }
};

void vector_add(const float* a, const float* b, float* out, size_t n){
    const float* ptr_a = a;
    const float* ptr_b = b;
    float* ptr_out = out;
    for(size_t i=0; i < n; i++){
        *ptr_out = *ptr_a * *ptr_b;
        ptr_a++;
        ptr_b++;
        ptr_out++;
    }
}

void process_vector(const float* data, size_t size){
    if(size > 0) std::cout << data[0] << std::endl;
}

int main(){
    std::unique_ptr<VectorBuffer> vector_buffer = std::make_unique<VectorBuffer>(1024);
    process_vector(vector_buffer->d_ptr, vector_buffer->size);
    return 0;
}