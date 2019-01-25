# TODO: CURRENTLY WIP


from google.cloud import pubsub
from lib_learning.collection.interfaces.base_interface import Interface


class PubSubInterface(Interface):
    def __init__(self, pipeline_name, project_id, logger):
        self.pipeline_name = pipeline_name
        self.project_id = project_id
        self.logger = logger

        self.status_key = '{}_status'.format(self.pipeline_name)
        self.work_key = '{}_work'.format(self.pipeline_name)

        self.publisher = pubsub.PublisherClient()
        self.work_topic_path = self.publisher.topic_path(project_id, self.work_key)
        self.status_topic_path = self.publisher.topic_path(project_id, self.status_key)

        self.subscriber = pubsub.SubscriberClient()
        self.work_sub_path = 'projects/{}/subscriptions/{}'.format(self.project_id, self.work_key)
        self.status_sub_path = 'projects/{}/subscriptions/{}'.format(self.project_id, self.status_key)

        self.init_topics()
        self.init_subscriptions()


    def init_topics(self):
        existing_topics = self.publisher.list_topics(self.publisher.project_path(self.project_id))
        existing_topics = {topic.name for topic in existing_topics}

        if self.work_topic_path not in existing_topics:
            self.publisher.create_topic(self.work_topic_path)

        if self.status_topic_path not in existing_topics:
            self.publisher.create_topic(self.status_topic_path)


    def init_subscriptions(self):
        existing_subs = self.subscriber.list_subscriptions(self.subscriber.project_path(self.project_id))
        existing_subs = {subs.name for subs in existing_subs}

        if self.work_sub_path not in existing_subs:
            self.subscriber.create_subscription(self.work_sub_path, self.work_topic_path)

        if self.status_sub_path not in existing_subs:
            self.subscriber.create_subscription(self.status_sub_path, self.status_topic_path)


    def push_work(self):
        pass


    def recieve_work(self):
        pass


    def push_confirmation(self):
        pass


    def recieve_confirmation(self):
        pass
