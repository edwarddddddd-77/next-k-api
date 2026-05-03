# 供 s3_accumulation_radar 使用：币安广场热搜（6H）列表。
# 若后续接入真实接口，请返回 [{"coin": "BTC", "rapidRiser": False}, ...]，按热度排序。


def get_square_heat():
    """广场热搜；未接入数据源时返回空列表，其余逻辑仍可用 CG + 放量。"""
    return []
