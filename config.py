"""Configuration schema for github.yunshanluan0714.social-intuition v1.0.0."""

from maibot_sdk import Field, PluginConfigBase


class PluginSection(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="启用熟人社交直觉套件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class FollowupSection(PluginConfigBase):
    __ui_label__ = "生活近况追问"
    __ui_icon__ = "heart"
    __ui_order__ = 1

    enabled: bool = Field(default=True, description="静默记住群友提到的近期要事，事后冒泡时自然追问后续")
    use_llm_verification: bool = Field(default=True, description="使用小模型复核是否为值得记挂的生活事件并精修追问时间")
    llm_task_name: str = Field(default="fast_model", description="复核使用的模型任务路由名")
    min_confidence: float = Field(default=0.75, ge=0.3, le=0.99, description="最低入库置信度")
    expire_days: int = Field(default=5, ge=1, le=30, description="超过到期时间几天后自动过期不再硬问")
    user_cooldown_days: int = Field(default=7, ge=1, le=60, description="同一群内对同一个人两次追问的最小冷却天数")
    max_pending_per_user: int = Field(default=3, ge=1, le=10, description="单群每人最多同时保留的待追问近况条数")


class DejavuSection(PluginConfigBase):
    __ui_label__ = "跨月既视感"
    __ui_icon__ = "repeat"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="发现某位熟人再次发出跟两周/数月前高度相似的吐槽时顺嘴翻旧账")
    min_days_apart: int = Field(default=14, ge=3, le=365, description="两次相似发言至少间隔多少天（可按群历史调高至30）")
    use_embedding: bool = Field(default=True, description="启用向量语义相似度（失败或无向量时自动退化到字面相似）")
    embedding_task_name: str = Field(default="embedding", description="向量嵌入任务名")
    embedding_threshold: float = Field(default=0.86, ge=0.7, le=0.99, description="向量余弦相似度触发阈值")
    lexical_threshold: float = Field(default=0.42, ge=0.25, le=0.9, description="字符 2-gram 相似度兜底阈值")
    user_cooldown_days: int = Field(default=14, ge=1, le=90, description="同人两次触发既视感的最小冷却天数")
    max_utterances_per_group: int = Field(default=4000, ge=500, le=20000, description="单群最多保存的历史观点句条数")


class ReunionSection(PluginConfigBase):
    __ui_label__ = "失踪老友回归"
    __ui_icon__ = "user-check"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="平时活跃的熟人彻底潜水半月以上再冒泡时自然打声招呼")
    min_absent_days: int = Field(default=15, ge=3, le=180, description="至少失踪多少天不发言才视为回归")
    min_history_messages: int = Field(default=10, ge=3, le=200, description="历史至少说过多少句话才算老熟人（过滤路人）")
    cooldown_days: int = Field(default=30, ge=7, le=365, description="同一群对同一位老友触发回归感应的冷却天数")


class GuardSection(PluginConfigBase):
    __ui_label__ = "隐私、静默与历史回填"
    __ui_icon__ = "shield"
    __ui_order__ = 4

    allow_private: bool = Field(default=True, description="是否同时在私聊各自独立生效")
    allowed_groups: list[str] = Field(default_factory=list, description="仅在指定群号列表生效（留空为全部群）")
    blocked_groups: list[str] = Field(default_factory=list, description="黑名单群号列表")
    quiet_start: str = Field(default="01:00", description="深夜免打扰开始时间（HH:MM），只记账不注入提示")
    quiet_end: str = Field(default="08:00", description="深夜免打扰结束时间（HH:MM）")
    timezone: str = Field(default="Asia/Shanghai", description="时区")
    backfill_on_startup: bool = Field(default=True, description="首次加载时低速只读回填各群近60天历史以便首日即可感应")
    backfill_days: int = Field(default=60, ge=7, le=180, description="首次回填的历史天数范围")


class SocialIntuitionConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    followup: FollowupSection = Field(default_factory=FollowupSection)
    dejavu: DejavuSection = Field(default_factory=DejavuSection)
    reunion: ReunionSection = Field(default_factory=ReunionSection)
    guard: GuardSection = Field(default_factory=GuardSection)
