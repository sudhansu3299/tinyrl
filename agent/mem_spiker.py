import time
def simulate_memory_spike(chunk_size_mb=2):
    print("🚨 Starting memory spike simulation")
    memory_hog = []
    for i in range(300):
        memory_hog.append(bytearray(chunk_size_mb * 1024 * 1024))  # Allocate real memory
        print(f"📈 Allocated {(i+1)*chunk_size_mb} MB")
        time.sleep(1)
    print("Finished memory spike simulation!")

simulate_memory_spike()