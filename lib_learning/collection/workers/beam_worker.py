import apache_beam as beam


p = beam.Pipeline(argv=['--streaming'])
lines = (
    p |
    'pubsub_read' >> beam.io.ReadFromPubSub(topic='projects/earth-229521/topics/test_pipeline_work') |
    'window' >> beam.WindowInto(beam.window.FixedWindows(5)) |
    'file_write' >> beam.io.WriteToText('test_file')
)
p.run()
