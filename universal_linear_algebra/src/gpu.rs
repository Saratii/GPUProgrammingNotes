use std::mem::size_of_val;
use wgpu::{util::DeviceExt, DeviceDescriptor, Limits};

pub struct GPU {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GPU {
    pub async fn new() -> Self {
        let limits = Limits {
            max_buffer_size: 400_000_000,           
            max_storage_buffer_binding_size: 400_000_000,
            ..Limits::default()
        };
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect("Failed to find an appropriate adapter");
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_limits: limits,
                ..DeviceDescriptor::default()
            }, None)
            .await
            .unwrap();
        GPU { device, queue }
    }

    pub async fn add_vecs(&self, numbers1: &[f32], numbers2: &[f32]) -> Vec<f32> {
        let (staging_buffer, storage_buffer1, compute_pipeline, bind_group) = self.setup_pipeline(numbers1, numbers2);
        self.dispatch_gpu_commands(&storage_buffer1, &staging_buffer, &compute_pipeline, &bind_group, numbers1.len());
        self.retrieve_results(&staging_buffer).await
    }

    fn dispatch_gpu_commands(
        &self,
        storage_buffer1: &wgpu::Buffer,
        staging_buffer: &wgpu::Buffer,
        compute_pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        num_elements: usize,
    ) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
               ..Default::default()
            });
            cpass.set_pipeline(compute_pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups( ((num_elements + 255) / 256) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(storage_buffer1, 0, staging_buffer, 0, size_of_val(&[0f32; 1]) as wgpu::BufferAddress * num_elements as wgpu::BufferAddress);
        self.queue.submit(Some(encoder.finish()));
    }

    fn create_storage_buffer(&self, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn create_staging_buffer(&self, size: wgpu::BufferAddress) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    fn setup_pipeline(
        &self,
        numbers1: &[f32],
        numbers2: &[f32],
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::ComputePipeline,
        wgpu::BindGroup,
    ) {
        let cs_module = self.device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let size = size_of_val(numbers1) as wgpu::BufferAddress;
        let staging_buffer = self.create_staging_buffer(size);
        let storage_buffer1 = self.create_storage_buffer(numbers1);
        let storage_buffer2 = self.create_storage_buffer(numbers2);
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer2.as_entire_binding(),
                },
            ],
        });
        (staging_buffer, storage_buffer1, compute_pipeline, bind_group)
    }

    async fn retrieve_results(&self, staging_buffer: &wgpu::Buffer) -> Vec<f32> {
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            return result
        } 
        panic!("failed to run compute on gpu!")
    }
}