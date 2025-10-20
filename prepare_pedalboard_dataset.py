from pedalboard import Pedalboard, Reverb, load_plugin, VST3Plugin
from pedalboard.io import AudioFile
from mido import Message 
import random
import re
import threading
import multiprocessing as mp
import time
import os
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor
import pickle
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

def randomize_params(inst, params):
    for k, v in params.items():
        if None not in v.range:
            low, high, step = v.range
            value = random.uniform(low, high)
            if v.type == bool:
                value = random.choice([True, False])
                # if 'enable' in k:
                #     value = True
            # if 'level' in k:
            #     value = 0.5
            setattr(inst, k, value)
        else:
            valid_values = v.valid_values
            value = valid_values[random.randint(0, len(valid_values) - 1)]
            # if 'enable' in k:
            #     value = True
            if 'level' in k:
                value = valid_values[-1]
            setattr(inst, k, value)

def get_target_params(inst):
    list_params = inst.parameters
    list_params_ = {}
    for k, v in list_params.items():
        if k.startswith('a_') or k.startswith('b_') or k.startswith('c_') \
            or k.startswith('filter_1') or k.startswith('filter_2') \
            or k.startswith('sub_'):
            list_params_[k] = v
    return list_params_

def dump_raw_values(params):
    raw_values = {}
    for k, v in params.items():
        raw_values[k] = v.raw_value
    return raw_values

# Standalone worker function for multiprocessing (must be picklable)
def audio_generation_worker_func(worker_id, audio_queue, stop_event, generated_count, lock, total_samples, sample_rate, audio_duration):
    """Worker process for generating audio with randomized parameters"""
    try:
        # Set CPU affinity to bind this process to specific cores (Windows)
        try:
            import psutil
            p = psutil.Process()
            # Distribute workers across available CPU cores
            cpu_count = psutil.cpu_count(logical=True)
            # Assign cores in a round-robin fashion
            core_id = worker_id % cpu_count
            p.cpu_affinity([core_id])
            print(f'Worker {worker_id}: Bound to CPU core {core_id}')
        except Exception as e:
            print(f'Worker {worker_id}: Could not set CPU affinity: {e}')
        
        print(f'Worker {worker_id}: Initializing plugin...')
        inst = load_plugin(
            "C:\\Program Files\\Common Files\\vst3\\Serum2.vst3\\Contents\\x86_64-win\\Serum2.vst3", 
            plugin_name="Serum 2",
        )
        print(f'Worker {worker_id}: Plugin loaded')
        target_params = get_target_params(inst)
        
        while not stop_event.is_set():
            try:
                # Check if we've reached the target
                with lock:
                    if generated_count.value >= total_samples:
                        print(f"Worker {worker_id}: Reached target samples, stopping")
                        break
                
                # Generate random parameters
                randomize_params(inst, target_params)
                
                # Generate audio
                audio = inst(
                    [Message("note_on", note=60), Message("note_off", note=60, time=2)],
                    duration=audio_duration,
                    sample_rate=sample_rate,
                )
                
                # Get parameter values
                param_values = dump_raw_values(target_params)
                
                # Convert numpy array to ensure serialization
                audio_numpy = np.array(audio, dtype=np.float32)
                
                # Put in queue for embedding
                audio_data = {
                    'audio': audio_numpy,
                    'params': param_values,
                    'worker_id': worker_id
                }
                
                audio_queue.put(audio_data, timeout=5)
                
                with lock:
                    generated_count.value += 1
                        
            except Exception as e:
                if stop_event.is_set():
                    break
                print(f"Worker {worker_id}: Error generating audio: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
                
    except Exception as e:
        print(f"Worker {worker_id}: Failed to initialize: {e}")
        import traceback
        traceback.print_exc()

class DatasetGenerator:
    def __init__(self, output_dir="output", num_workers=4, chunk_size=10000, total_samples=1000000):
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.total_samples = total_samples
        # Use multiprocessing queues for inter-process communication
        self.audio_queue = mp.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.sample_rate = 48000
        self.audio_duration = 2.0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Multiprocessing control - use Manager for shared state
        self.manager = mp.Manager()
        self.stop_event = self.manager.Event()
        self.generated_count = self.manager.Value('i', 0)
        self.embedded_count = self.manager.Value('i', 0)
        self.saved_count = self.manager.Value('i', 0)
        
        # Progress tracking
        self.lock = self.manager.Lock()
        
    def get_existing_chunks(self):
        """Get list of existing chunk files to support resumability"""
        existing_chunks = []
        for filename in os.listdir(self.output_dir):
            if filename.startswith("chunk_") and filename.endswith(".pkl"):
                chunk_num = int(filename.split("_")[1].split(".")[0])
                existing_chunks.append(chunk_num)
        return sorted(existing_chunks)
    
    def embedding_worker(self):
        """Single GPU worker for audio embedding (runs as thread)"""
        try:
            print('Embedding worker: Loading CLAP model...')
            model = ClapModel.from_pretrained("laion/larger_clap_music")
            processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
            model = model.cuda()
            model.eval()
            print('Embedding worker: Model loaded on GPU')
            
            current_chunk = []
            chunk_num = len(self.get_existing_chunks())
            
            with torch.no_grad():
                while not self.stop_event.is_set() or not self.audio_queue.empty():
                    try:
                        # Get audio data from queue
                        audio_data = self.audio_queue.get(timeout=2)
                        
                        # Process audio through CLAP
                        inputs = processor(audios=audio_data['audio'], return_tensors="pt")
                        inputs['input_features'] = inputs['input_features'].cuda()
                        inputs['sampling_rate'] = 48000
                        audio_embed = model.get_audio_features(**inputs)
                        
                        # Convert to bf16 and move to CPU for storage
                        audio_embed_bf16 = audio_embed.to(torch.bfloat16).cpu()
                        
                        # Create data sample
                        sample = {
                            'features': audio_embed_bf16,
                            'params': audio_data['params'],
                            'worker_id': audio_data['worker_id']
                        }
                        
                        current_chunk.append(sample)
                        
                        with self.lock:
                            self.embedded_count.value += 1
                        
                        # Save chunk when it reaches the desired size
                        if len(current_chunk) >= self.chunk_size:
                            self.save_chunk(current_chunk, chunk_num)
                            current_chunk = []
                            chunk_num += 1
                        
                    except Exception as e:
                        if self.stop_event.is_set():
                            break
                        if "Empty" not in str(type(e).__name__): b
                            print(f"Embedding worker: Error processing audio: {e}")
                        continue
            
            # Save remaining samples
            if current_chunk:
                self.save_chunk(current_chunk, chunk_num)
                
        except Exception as e:
            print(f"Embedding worker: Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
    
    def save_chunk(self, chunk_data, chunk_num):
        """Save a chunk of data to disk"""
        try:
            chunk_path = os.path.join(self.output_dir, f"chunk_{chunk_num:06d}.pkl")
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_data, f)
            
            with self.lock:
                self.saved_count.value += len(chunk_data)
                current_saved = self.saved_count.value
            
            print(f"Saved chunk {chunk_num} with {len(chunk_data)} samples. Total saved: {current_saved}")
            
            # Save metadata
            metadata = {
                'chunk_num': chunk_num,
                'chunk_size': len(chunk_data),
                'total_saved': current_saved,
                'timestamp': time.time()
            }
            
            metadata_path = os.path.join(self.output_dir, f"chunk_{chunk_num:06d}_meta.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving chunk {chunk_num}: {e}")
    
    def progress_monitor(self):
        """Monitor and display progress"""
        pbar = tqdm(total=self.total_samples, desc="Generating dataset")
        last_saved = 0
        
        while not self.stop_event.is_set() and self.saved_count.value < self.total_samples:
            time.sleep(2)
            with self.lock:
                current_saved = self.saved_count.value
                generated = self.generated_count.value
                embedded = self.embedded_count.value
            
            pbar.update(current_saved - last_saved)
            pbar.set_postfix({
                'Generated': generated,
                'Embedded': embedded,
                'Saved': current_saved,
                'AudioQ': self.audio_queue.qsize()
            })
            last_saved = current_saved
        
        pbar.close()
    
    def generate_dataset(self):
        """Main function to orchestrate dataset generation"""
        print(f"Starting dataset generation:")
        print(f"- Target samples: {self.total_samples}")
        print(f"- Chunk size: {self.chunk_size}")
        print(f"- Workers: {self.num_workers}")
        print(f"- Output directory: {self.output_dir}")
        
        # Check existing chunks
        existing_chunks = self.get_existing_chunks()
        if existing_chunks:
            print(f"Found {len(existing_chunks)} existing chunks. Resuming from chunk {max(existing_chunks) + 1}")
            self.saved_count.value = len(existing_chunks) * self.chunk_size
        
        try:
            # Start worker processes and threads
            processes = []
            threads = []
            
            # Start audio generation workers as separate processes
            for i in range(self.num_workers):
                process = mp.Process(
                    target=audio_generation_worker_func, 
                    args=(i, self.audio_queue, self.stop_event, self.generated_count, 
                          self.lock, self.total_samples, self.sample_rate, self.audio_duration)
                )
                process.start()
                processes.append(process)
            
            # Start embedding worker as a thread (needs GPU access)
            embed_thread = threading.Thread(target=self.embedding_worker)
            embed_thread.daemon = True
            embed_thread.start()
            threads.append(embed_thread)
            
            # Start progress monitor as a thread
            progress_thread = threading.Thread(target=self.progress_monitor)
            progress_thread.daemon = True
            progress_thread.start()
            threads.append(progress_thread)
            
            # Wait for completion or interruption
            try:
                while self.saved_count.value < self.total_samples:
                    time.sleep(1)
                    # Check if all processes are still alive
                    if all(not p.is_alive() for p in processes):
                        print("All worker processes finished")
                        break
            except KeyboardInterrupt:
                print("\nInterrupting dataset generation...")
                self.stop_event.set()
            
            print("Stopping workers...")
            self.stop_event.set()
            
            # Wait for processes to finish
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    print(f"Process {process.pid} didn't stop, terminating...")
                    process.terminate()
                    process.join(timeout=2)
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=10)
            
            print(f"\nDataset generation completed!")
            print(f"Generated: {self.generated_count.value} samples")
            print(f"Embedded: {self.embedded_count.value} samples") 
            print(f"Saved: {self.saved_count.value} samples")
            
        except Exception as e:
            print(f"Error during dataset generation: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()

def main():
    """Main entry point"""
    # Set multiprocessing start method (important for Windows)
    mp.set_start_method('spawn', force=True)
    
    generator = DatasetGenerator(
        output_dir="output",
        num_workers=10,  # Adjust based on your CPU cores
        chunk_size=10000,
        total_samples=1000000
    )
    
    generator.generate_dataset()

if __name__ == "__main__":
    # This guard is required for multiprocessing on Windows
    main()
