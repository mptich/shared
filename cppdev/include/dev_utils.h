#ifndef __DEV_UTILS_H__
#define __DEV_UTILS_H__

typedef void (*StaticCallClassCallback)(void);

class StaticCallClass {
public:
	StaticCallClass (StaticCallClassCallback c) { (*c)(); }
};

#define StaticCall(x) \
	StaticCallClass __stat_call_class_ ## x (x);

#endif
