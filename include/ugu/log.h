/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

namespace ugu {

enum class LogLevel {
  kVerbose = 0,
  kDebug = 1,
  kInfo = 2,
  kWarning = 3,
  kError = 4,
  kNone = 5
};

void set_log_level(LogLevel level);
LogLevel get_log_level();
// To avoid conflict with macro
#ifndef LOGD
void LOGD(const char *format, ...);
#endif
#ifndef LOGI
void LOGI(const char *format, ...);
#endif
#ifndef LOGW
void LOGW(const char *format, ...);
#endif
#ifndef LOGE
void LOGE(const char *format, ...);
#endif

}  // namespace ugu
